from niche_bin import bin_to_raw
import numpy as np
from pathlib import Path
from scipy.signal import argrelextrema

from config import NICHE_TIMEBIN_SIZE, WAVEFORM_SIZE, N_SIM_TRIGGER_WINDOWS
from utils import read_noise_file

def noise_fft(noise_array: np.ndarray) -> np.ndarray:
    '''This function returns the Fourier transform of an array of noise data
    Parameters:
    noise_array: array of noise data
    returns fft
    '''
    ft = np.abs(np.fft.fft(noise_array))
    return ft.mean(axis=0)

def freq_mhz() -> np.ndarray:
    '''This function returns the frequency domain for the fft of a niche noise trace
    in MHz.
    '''
    return np.fft.fftfreq(WAVEFORM_SIZE, NICHE_TIMEBIN_SIZE * 1.e-9) * 1.e-6

def increment_phases(phase_angles: np.ndarray, cutoff_freq: float = 10.e6) -> np.ndarray:
    '''This function takes the phase angles of the high frequency modes and randomizes them 
    (this is the equivalent of incrementing the phases an integer number of cycles during 
    the time of a trace) and keepts the low frequency phases the same.
    '''
    new_phases = np.empty_like(phase_angles)
    freq = freq_mhz() * 1.e6 #in Hz
    pos_freq = freq[freq>0]
    new_phases[pos_freq<cutoff_freq] = phase_angles[pos_freq<cutoff_freq]
    new_phases[pos_freq>=cutoff_freq] = np.random.rand(new_phases[pos_freq>=cutoff_freq].size) * 2 * np.pi
    # new_phases = np.random.rand(phase_angles.size) * 2 * np.pi
    return new_phases

def random_noise_snapshot(ft: np.ndarray, phase_angles: np.ndarray) -> np.ndarray:
    '''This function generates a random noise snapshot with a given power spectrum (ft)
    and phase angles.
    '''
    n_phases = phase_angles.size
    ft_complex = np.array(ft, dtype='complex')
    complex_phases = np.cos(phase_angles) + 1j * np.sin(phase_angles)
    ft_complex[1:n_phases+1] *= complex_phases
    ft_complex[-1:-1-n_phases:-1] = np.conj(ft_complex[1:n_phases+1])
    return np.abs(np.fft.ifft(ft_complex))

def shift_phase_angles(phase_angles: np.ndarray, term: int) -> np.ndarray:
    '''This function takes the phase angles and shifts them so they are relative
    to the term index specified by term.
    '''
    shift = np.arange(phase_angles.shape[1])*phase_angles[:,term][:,np.newaxis]/term
    return phase_angles - shift

def phase_shift(phase_angles: np.ndarray, term: int) -> np.ndarray:
    return np.arange(phase_angles.shape[1])*phase_angles[:,term][:,np.newaxis]/term

def random_phases_from_ecdf(phase_angles: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    '''This function takes relative phase angles, makes an ecdf of the dist of
    each term, and draws a corresponding random value from each.
    '''
    cdf_values = np.arange(phase_angles.shape[0])/phase_angles.shape[0]
    random_phases = np.empty(phase_angles.shape[1])
    corresponding_shifts = np.empty_like(random_phases)
    thrown_cfd_values = np.random.rand(phase_angles.shape[1])
    for i, val in enumerate(thrown_cfd_values):
        random_phases[i] = np.interp(val, cdf_values, np.sort(phase_angles[:,i]))
        corresponding_shifts[i] = np.interp(val, cdf_values, shifts[:,i][np.argsort(phase_angles[:,i])])
    return random_phases, corresponding_shifts

def random_values_from_ecdf(data_values: np.ndarray, N_traces: int = 1) -> np.ndarray:
    '''This function takes a 2d array of data from a noise file, where the first index
    is the index of the trace, and the second index is that of the data within the 
    trace. In this context the data is either the power spectrum of each trace,
    or the adjusted phase angles, both from the fft.
    '''
    cdf_values = np.arange(data_values.shape[0])/data_values.shape[0]
    generated_data = np.empty((N_traces, data_values.shape[1]))
    thrown_cfd_values = np.random.rand(*generated_data.shape)
    for i in range(data_values.shape[1]):
        generated_data[:,i] = np.interp(thrown_cfd_values[:,i], cdf_values, np.sort(data_values[:,i]))
    return generated_data

def calc_epdf_bins(real_fadc_array: np.ndarray, sim_fadc_array: np.ndarray) -> np.ndarray:
    '''Calculate bins for the histogram of all real and initially simulated noise data.
    '''
    both = np.hstack((real_fadc_array, sim_fadc_array))
    min = np.min(both)
    max = np.max(both)
    midbins = np.arange(min, max + 1)
    bins = midbins + .5
    return midbins[1:], bins

def draw_from_discrete_dist(vals: np.ndarray, probs: np.ndarray, N: int = 1) -> np.ndarray:
    '''Draw N random values from a discretely defined pdf.
    '''
    dcdf_values = np.cumsum(probs) / probs.sum()
    thrown_dcdf_values = np.random.rand(N)
    return np.round(np.interp(thrown_dcdf_values,dcdf_values,vals)).astype(int)
    # thrown_values = np.empty(N)
    # thrown_values = np.array([vals[np.argmax(val<=dcdf_values)] for val in thrown_dcdf_values])
    # return thrown_values

def add_glitches(real_fadc_array: np.ndarray, sim_fadc_array: np.ndarray, cut_prob: float = 0) -> np.ndarray:
    '''This is the procedure for adding the high and low value fadc count glitches which are present in 
    real noise traces but not reproduced by Fourier analysis.
    '''
    sim_with_glitches = np.copy(sim_fadc_array)
    fadc_values, bins = calc_epdf_bins(real_fadc_array, sim_fadc_array)
    prob_in_real, _ = np.histogram(real_fadc_array, bins = bins, density = True)
    prob_in_sim, _ = np.histogram(sim_fadc_array, bins = bins, density = True)
    diff = prob_in_real - prob_in_sim
    prob_different = np.abs(diff)
    up_for_change_mask = np.full(sim_fadc_array.shape, False, dtype=bool)
    for fadc in fadc_values[diff<=cut_prob]:
        up_for_change_mask += sim_fadc_array == fadc
    up_for_change = sim_fadc_array[up_for_change_mask]
    N_to_change = int(np.round(prob_different.sum() * up_for_change.size))
    # N_to_change = int(np.round(diff[diff>0.].sum() * up_for_change.size))
    if N_to_change == 0:
        return sim_with_glitches
    else:
        ch_mask = np.random.randint(0,up_for_change.size,size=N_to_change)
        diff[diff<0.] = 0.
        up_for_change[ch_mask] = draw_from_discrete_dist(fadc_values, diff, N = N_to_change)
        sim_with_glitches[up_for_change_mask] = up_for_change
        return sim_with_glitches

def random_noise(noisefile: Path, N_windows: int = 1, cut_prob: float = 0) -> tuple[np.ndarray,np.ndarray]:
    '''This function is the procedure for generating a simulated noise trace 
    for a given noise file.
    '''
    #Read noise file and take its ft
    noise_array = read_noise_file(noisefile)
    ft = np.fft.fft(noise_array)
    freqs = freq_mhz()
    phase_angles = np.angle(ft)

    #Create output array
    noise_output =np.empty(WAVEFORM_SIZE*N_windows)

    #Find index of murmur mode
    im = argrelextrema(np.abs(ft).mean(axis=0), np.greater)[0][0]

    #Shift phase angles so the murmur term in ft is zero phase
    shifts = phase_shift(phase_angles,im)
    shifted_angles = phase_angles - shifts

    #Generate random phases from ecdfs created from shifted phases in data, unshift
    random_phases, unshifts = random_phases_from_ecdf(shifted_angles, shifts)
    # random_phases = np.random.rand(shifted_angles.shape[1]) *2 * np.pi
    # random_phases += unshifts

    #Generate random power spectrum values from ecdfs created from data
    gen_ft = random_values_from_ecdf(np.abs(ft),N_windows)

    #Take only the phases for the positive frequencies
    random_phases = random_phases[freqs>0]
    #Loop through the desired number of trigger windows to stack
    for i in range(N_windows):
        starti = i*WAVEFORM_SIZE
        #dump inverse fft of random phases and power spectrum into output array
        noise_output[starti:starti + WAVEFORM_SIZE] = random_noise_snapshot(gen_ft[i], random_phases)
    
    fadc_counts = np.round(noise_output).astype('int')
    fadc_counts_glitched = add_glitches(noise_array.flatten(),fadc_counts,cut_prob=cut_prob)
    return fadc_counts_glitched, fadc_counts

# class NoiseGen:
#     '''This class save the fft of the noise so random traces can be repeatedly generated.
#     '''
#     def __init__(self,noisefile: Path) -> None:
#         self.noisefile = noisefile
#         #Read noise file and take its ft
#         self.noise_array = read_noise_file(noisefile)
#         self.ft = np.fft.fft(self.noise_array)
#         self.freqs = freq_mhz()
#         self.phase_angles = np.angle(self.ft)
#         #Find index of murmur mode
#         self.im = argrelextrema(np.abs(self.ft).mean(axis=0), np.greater)[0][0]
#         #Shift phase angles so the murmur term in ft is zero phase
#         self.shifts = phase_shift(self.phase_angles,self.im)
#         self.shifted_angles = self.phase_angles - self.shifts
    
#     def random_noise(self, N_windows: int = 1, cut_prob: float = 0) -> tuple[np.ndarray,np.ndarray]:
#         '''This function is the procedure for generating a simulated noise trace 
#         for the class' noise file.
#         '''
#         #Create output array
#         noise_output =np.empty(WAVEFORM_SIZE*N_windows)

#         #Generate random phases from ecdfs created from shifted phases in data, unshift
#         random_phases, unshifts = random_phases_from_ecdf(self.shifted_angles, self.shifts)
#         # random_phases = np.random.rand(shifted_angles.shape[1]) *2 * np.pi
#         # random_phases += unshifts

#         #Generate random power spectrum values from ecdfs created from data
#         gen_ft = random_values_from_ecdf(np.abs(self.ft),N_windows)

#         #Take only the phases for the positive frequencies
#         random_phases = random_phases[self.freqs>0]
#         #Loop through the desired number of trigger windows to stack
#         for i in range(N_windows):
#             starti = i*WAVEFORM_SIZE
#             #dump inverse fft of random phases and power spectrum into output array
#             noise_output[starti:starti + WAVEFORM_SIZE] = random_noise_snapshot(gen_ft[i], random_phases)
        
#         fadc_counts = np.round(noise_output).astype('int')
#         fadc_counts_glitched = add_glitches(self.noise_array.flatten(),fadc_counts,cut_prob=cut_prob)
#         return fadc_counts_glitched, fadc_counts

class NoiseGen:
    '''This class save the fft of the noise so random traces can be repeatedly generated.
    '''
    def __init__(self,noisefile: Path) -> None:
        self.noisefile = noisefile
        self.long_trace, _ = random_noise(noisefile, 10*N_SIM_TRIGGER_WINDOWS)

    def random_noise(self, N_windows: int = 1) -> tuple[np.ndarray,None]:
        '''This function is the procedure for generating a simulated noise trace 
        for the class' noise file.
        '''
        return_trace_length = N_windows*WAVEFORM_SIZE
        maxi = len(self.long_trace) - return_trace_length
        starti = np.random.randint(0,maxi)
        return self.long_trace[starti:starti+return_trace_length], None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    p_open = Path('/home/isaac/niche_data/20190509/bell/20190509060703.bg.bin')
    p_closed = Path('/home/isaac/niche_data/20190509/bell/20190509060628.bg.bin')

    gen_open = NoiseGen(p_open)
    gen_closed = NoiseGen(p_closed)

    noise_open = read_noise_file(p_open)
    noise_closed = read_noise_file(p_closed)
    ft_o = noise_fft(noise_open)
    ft_c = noise_fft(noise_closed)
    freq = freq_mhz()

    sim_noise,_ = random_noise(p_open)
    sim_noise_closed,_ = random_noise(p_closed)

    gen_sim_noise,_ = gen_open.random_noise()
    gen_sim_noise_closed,_ = gen_closed.random_noise()

    ft_s = np.abs(np.fft.fft(sim_noise))
    ft_sc = np.abs(np.fft.fft(sim_noise_closed))

    gen_ft_s = np.abs(np.fft.fft(gen_sim_noise))
    gen_ft_sc = np.abs(np.fft.fft(gen_sim_noise_closed))
    # r = random_noise(p_open,5)

    plt.figure()
    plt.plot(freq[freq>0.], ft_o[freq>0.], label = 'noise open', c='k', linestyle='solid')
    plt.plot(freq[freq>0.], ft_c[freq>0.], label = 'noise closed', c='k', linestyle='dashed')

    # plt.plot(freq[freq>0.], ft_s[freq>0.], label = 'noise sim')
    # plt.plot(freq[freq>0.], ft_sc[freq>0.], label = 'noise sim closed')
    # plt.plot(freq[freq>0.], gen_ft_s[freq>0.], label = 'gen noise sim')
    # plt.plot(freq[freq>0.], gen_ft_sc[freq>0.], label = 'gen noise sim closed')
    
    n_to_avg = 4096
    prob= 0.
    traces_to_avg = np.empty((n_to_avg,WAVEFORM_SIZE))
    long_trace,_ = gen_open.random_noise(N_windows=n_to_avg,cut_prob=prob)
    # long_trace,_ = random_noise(p_open, N_windows=n_to_avg,cut_prob=prob)
    traces_to_avg_c = np.empty((n_to_avg,WAVEFORM_SIZE))
    long_trace_c,_ = gen_closed.random_noise(N_windows=n_to_avg,cut_prob=prob)
    # long_trace_c,_ = random_noise(p_closed, N_windows=n_to_avg,cut_prob=prob)
    for i in range(traces_to_avg.shape[0]):
        starti = i*WAVEFORM_SIZE
        # starti = np.random.randint(0,long_trace_c.size-WAVEFORM_SIZE)
        traces_to_avg[i] = np.abs(np.fft.fft(long_trace[starti:starti + WAVEFORM_SIZE]))
        traces_to_avg_c[i] = np.abs(np.fft.fft(long_trace_c[starti:starti + WAVEFORM_SIZE]))
        # plt.plot(freq[freq>0], np.abs(np.fft.fft(r[i:i+1024]))[freq>0],label=f'start index: {i}')
    plt.plot(freq[freq>0.], traces_to_avg.mean(axis=0)[freq>0.], label = 'avg sim open', c='r', linestyle='solid')
    plt.plot(freq[freq>0.], traces_to_avg_c.mean(axis=0)[freq>0.], label = 'avg sim closed', c='r', linestyle='dashed')
    plt.loglog()
    plt.grid()
    # plt.semilogy()
    plt.legend()
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Relative Power')
    plt.suptitle('Comparing Real and Simulated Power Spectra')
    plt.title(f"Counter '{p_open.parent.name}' during {p_open.name[:8]} observation")

    fig, axs = plt.subplots(nrows=5, sharex=True)
    for i, ax in enumerate(axs[:-1]):
        ax.plot(noise_open[i])
        ax.set_title(f'Noise trace #{i}')
    axs[-1].plot(sim_noise)
    axs[-1].set_title('simulated noise')

    ft = np.fft.fft(noise_open)
    freqs = freq_mhz()
    phase_angles = np.angle(ft)

    #Find index of murmur mode
    im = argrelextrema(np.abs(ft).mean(axis=0), np.greater)[0][0]

    #Shift phase angles so the murmur term in ft is zero phase
    shifts = phase_shift(phase_angles,im)
    shifted_angles = phase_angles - shifts

    #Generate random phases from ecdfs created from shifted phases in data, unshift
    random_phases, unshifts = random_phases_from_ecdf(shifted_angles, shifts)

    nc = noise_open.flatten()
    # nc = noise_closed.flatten()
    ns,n_noglitch = random_noise(p_open,4096,prob)
    midbins, bins = calc_epdf_bins(nc,ns)
    plt.figure()
    dist_c,_,_  =plt.hist(nc,bins=bins,histtype='step',density=True,label='real',color='k')
    dist_s,_,_  =plt.hist(ns,bins=bins,histtype='step',density=True,label='simulated',color='r')
    dist_ng,_,_  =plt.hist(n_noglitch,bins=bins,histtype='step',density=True,label='inverse fft',color='b')
    plt.xlabel('FADC Count Value')
    plt.ylabel('Probability / Sample')
    plt.legend()
    plt.grid()
    plt.suptitle('Comparing Real and Simulated FADC Count Distribution (4096 traces)')
    plt.title(f"Counter '{p_open.parent.name}' during {p_open.name[:8]} observation")
    plt.semilogy()