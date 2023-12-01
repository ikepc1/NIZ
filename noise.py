from niche_bin import bin_to_raw
import numpy as np
from pathlib import Path
from scipy.signal import argrelextrema

from config import NICHE_TIMEBIN_SIZE, WAVEFORM_SIZE

def read_niche_file(filepath: Path) -> np.ndarray:
    '''This function reads a noise file and returns a numoy array of the traces.
    '''
    with filepath.open('rb') as open_file:
        nraw_list = list(set(bin_to_raw(open_file.read(), filepath.parent.name, retfit=False)))
    return np.vstack([nraw.waveform for nraw in nraw_list])

def trigger_times(filepath: Path) -> np.ndarray:
    '''This function reads a niche file and returns a numoy array of the trigger times.
    '''
    with filepath.open('rb') as open_file:
        nraw_list = list(set(bin_to_raw(open_file.read(), filepath.parent.name, retfit=False)))
    return np.array([nraw.trigtime for nraw in nraw_list])

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

def increment_phases(phase_angles: np.ndarray) -> np.ndarray:
    '''This function takes the phase angles of each frequency mode and adds the
    amount of cycles each would go through during a NICHE snapshot.
    '''
    new_phases = np.empty_like(phase_angles)
    freq = freq_mhz() * 1.e6 #in Hz
    pos_freq = freq[freq>0]
    snapshot_time = WAVEFORM_SIZE * NICHE_TIMEBIN_SIZE * 1.e-9 #in seconds
    n_cycles = pos_freq * snapshot_time
    new_phases[pos_freq<10.e6] = phase_angles[pos_freq<10.e6] + n_cycles[pos_freq<10.e6] * 2 * np.pi
    new_phases[pos_freq>=10.e6] = np.random.rand(new_phases[pos_freq>=10.e6].size) * 2 * np.pi
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

def random_phases_from_ecdf(phase_angles: np.ndarray) -> np.ndarray:
    '''This function takes relative phase angles, makes an ecdf of the dist of
    each term, and draws a corresponding random value from each.
    '''
    cdf_values = np.arange(phase_angles.shape[0])/phase_angles.shape[0]
    random_phases = np.empty(phase_angles.shape[1])
    thrown_cfd_values = np.random.rand(phase_angles.shape[1])
    for i, val in enumerate(thrown_cfd_values):
        random_phases[i] = np.interp(val, cdf_values, np.sort(phase_angles[:,i]))
    return random_phases

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

def random_noise(noisefile: Path, N_windows: int = 1) -> np.ndarray:
    '''This function is the procedure for generating a simulated noise trace 
    for a given noise file.
    '''
    #Read noise file and take its ft
    noise_array = read_niche_file(noisefile)
    ft = np.fft.fft(noise_array)
    freqs = freq_mhz()
    phase_angles = np.angle(ft)

    #Create output array
    noise_output =np.empty(WAVEFORM_SIZE*N_windows)

    #Find index of murmur mode
    im = argrelextrema(ft.mean(axis=0), np.greater)[0][0]

    #Shift phase angles so the murmur term in ft is zero phase
    shift = phase_shift(phase_angles,im)
    shifted_angles = phase_angles - shift

    #Generate random phases from ecdfs created from shifted phases in data, unshift
    random_phases = random_phases_from_ecdf(shifted_angles) + shift[0]

    #Generate random power spectrum values from ecdfs created from data
    gen_ft = random_values_from_ecdf(np.abs(ft),N_windows)

    #Take only the phases for the positive frequencies
    random_phases = random_phases[freqs>0]
    #Loop through the desired number of trigger windows to stack
    for i in range(N_windows):
        starti = i*WAVEFORM_SIZE
        #dump inverse fft of random phases and power spectrum into output array
        noise_output[starti:starti + WAVEFORM_SIZE] = random_noise_snapshot(gen_ft[i], random_phases)
        #increment phases of low frequency terms, randomize high frequency phases
        random_phases = increment_phases(random_phases)
    return np.round(noise_output).astype('int')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    p_open = Path('/home/isaac/niche_data/20190509/bell/20190509060703.bg.bin')
    p_closed = Path('/home/isaac/niche_data/20190509/bell/20190509060628.bg.bin')
    noise_open = read_niche_file(p_open)
    noise_closed = read_niche_file(p_closed)
    ft_o = noise_fft(noise_open)
    ft_c = noise_fft(noise_closed)
    freq = freq_mhz()

    sim_noise = random_noise(p_open)
    sim_noise_closed = random_noise(p_closed)

    

    ft_s = np.abs(np.fft.fft(sim_noise))
    ft_sc = np.abs(np.fft.fft(sim_noise_closed))
    r = random_noise(p_open,5)

    plt.figure()
    plt.plot(freq[freq>0.], ft_o[freq>0.], label = 'noise open')
    plt.plot(freq[freq>0.], ft_c[freq>0.], label = 'noise closed')
    # plt.plot(freq[freq>0.], ft_s[freq>0.], label = 'noise sim')
    # plt.plot(freq[freq>0.], ft_sc[freq>0.], label = 'noise sim closed')
    
    n_to_avg = 4096
    traces_to_avg = np.empty((n_to_avg,WAVEFORM_SIZE))
    long_trace = random_noise(p_open, n_to_avg)
    traces_to_avg_c = np.empty((n_to_avg,WAVEFORM_SIZE))
    long_trace_c = random_noise(p_closed, n_to_avg)
    for i in range(traces_to_avg.shape[0]):
        starti = i*WAVEFORM_SIZE
        traces_to_avg[i] = np.abs(np.fft.fft(long_trace[starti:starti + WAVEFORM_SIZE]))
        traces_to_avg_c[i] = np.abs(np.fft.fft(long_trace_c[starti:starti + WAVEFORM_SIZE]))
        # plt.plot(freq[freq>0], np.abs(np.fft.fft(r[i:i+1024]))[freq>0],label=f'start index: {i}')
    plt.plot(freq[freq>0.], traces_to_avg.mean(axis=0)[freq>0.], label = 'avg sim open')
    plt.plot(freq[freq>0.], traces_to_avg_c.mean(axis=0)[freq>0.], label = 'avg sim closed')
    plt.loglog()
    # plt.semilogy()
    plt.legend()
    plt.xlabel('MHz')
    plt.ylabel('power')

    fig, axs = plt.subplots(nrows=5, sharex=True)
    for i, ax in enumerate(axs[:-1]):
        ax.plot(noise_open[i])
        ax.set_title(f'Noise trace #{i}')
    axs[-1].plot(sim_noise)
    axs[-1].set_title('simulated noise')


