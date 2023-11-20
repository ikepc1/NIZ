import numpy as np
from dataclasses import dataclass
from scipy.signal import fftconvolve
from pathlib import Path

from tunka_fit import TunkaPMTPulse
from write_niche import CounterTrigger
from gen_ckv_signals import CherenkovOutput
from noise import random_noise
from config import CounterConfig, WAVEFORM_SIZE, N_SIM_TRIGGER_WINDOWS, NICHE_TIMEBIN_SIZE, PHOTONS_WINDOW_SIZE, PHOTONS_TIMEBIN_SIZE, WAVEFORM_SIZE, TRIGGER_WIDTH, TRIGGER_VARIANCE 

def calc_bins(og_time_bins: np.ndarray, new_bin_size: float) -> np.ndarray:
    '''This method calculates the time bins needed to fit photon temporal
    signal in bins of size: new_bin_size.

    parameters:
    og_time_bins: array of original time bin edges (ns)
    new_bin_size: size of the new bins (ns)

    returns:
    array of new bin edges
    '''
    nbins = np.floor_divide(og_time_bins.max() - og_time_bins.min(), new_bin_size)
    ns_bins = og_time_bins.min() + (new_bin_size * np.arange(nbins))
    return np.concatenate((ns_bins, [og_time_bins.max()]))

def sum_over_wavelengths(photons: np.ndarray) -> np.ndarray:
    '''Temp function to sum over all ckv wavelengths.
    '''
    return photons.sum(axis=1)

def array_size(n_counters: int) -> tuple:
    '''This method returns the size of the photons and time bins arrays.
    '''
    return (n_counters, PHOTONS_WINDOW_SIZE,)

def bin_medians(bins: np.ndarray) -> np.ndarray:
    '''This method returns the middle of all the time bins
    '''
    return bins[:-1]+(bins[1:]-bins[:-1])/2

def photon_time_bins() -> np.ndarray:
    '''This method returns the full array of photon time bins of size
    PHOTONS_TIMEBIN_SIZE used for the signal in each counter
    '''
    width = (PHOTONS_TIMEBIN_SIZE * PHOTONS_WINDOW_SIZE) / 2
    return np.arange(-width, width + PHOTONS_TIMEBIN_SIZE , PHOTONS_TIMEBIN_SIZE)

def bin_photons(photons: np.ndarray, original_times: np.ndarray) -> np.ndarray:
    '''This method calculates the histogram of the photon counts in the original
    time bins in the PHOTONS_TIMEBIN_SIZE time bins.
    '''
    # bins = calc_bins(original_times, PHOTONS_TIMEBIN_SIZE)
    bins = photon_time_bins()
    return np.histogram(original_times, bins=bins, weights=photons, density = False)

def gen_photon_signal(n_cerenk_photons: np.ndarray, original_time_bins: np.ndarray) -> tuple:
    '''This method takes previously generated Cherenkov waveforms and
    centers it within an array the size PHOTONS_WINDOW_SIZE.
    n_photons: a rank 2 numpy array of photon #s of shape (#counters, #time bins)
    original_time_bins: a rank 2 numpy array of photon arrival times of shape
    (#counters, #time bins).

    returns:
    array of photon numbers
    shape: (#counters, PHOTONS_WINDOW_SIZE)
    '''
    signal_array = np.zeros(array_size(n_cerenk_photons.shape[0]))
    for i, (nps, otb) in enumerate(zip(n_cerenk_photons, original_time_bins)):
        signal_array[i], bins = bin_photons(nps, otb)
    return signal_array, bins

def generate_background(noise_files: list[Path]) -> np.ndarray:
    '''This function takes the noise files for a given night and returns an array of 
    simulated random noise with the same power spectrum as the real noise for that data part.
    '''
    noise_array = np.empty((len(noise_files), WAVEFORM_SIZE*N_SIM_TRIGGER_WINDOWS))
    for i, file in enumerate(noise_files):
        noise_array[i] = random_noise(file, N_SIM_TRIGGER_WINDOWS)
    return noise_array

class TriggerSim:
    '''This class is the procedure for simulating a trigger in each active niche counter.
    '''

    def __init__(self, cfg: CounterConfig) -> None:
        self.cfg = cfg
        self.array_shape = (len(cfg.active_counters), WAVEFORM_SIZE)

    @staticmethod
    def tunka_convolve(pes: np.ndarray, times: np.ndarray, delay: float) -> np.ndarray:
        '''This method convolves the photoelectron signal of a counter with a
        normalized Tunka style pmt pulse.
        '''
        tunka = TunkaPMTPulse(t0=delay)
        return fftconvolve(pes, tunka.tunka_pdf(times), mode='same')
    
    @staticmethod
    def NICHE_bins(g_time_bins: np.ndarray) -> np.ndarray:
        '''This method converts the photon time bins to NICHE timebins.
        '''
        return calc_bins(g_time_bins, NICHE_TIMEBIN_SIZE)

    def quantum_efficiency(self, incident_photons: np.ndarray) -> np.ndarray:
        '''This method converts incident photons to photoelectrons headed toward
        the first dynode.
        '''
        return (np.array(list(self.cfg.quantum_efficiency.values())) * incident_photons.T).T

    def gen_electron_signal(self, incident_photons: np.ndarray, times: np.ndarray) -> np.ndarray:
        '''This method converts the incident photons to each counter to
        electrons at each anode.
        '''
        anode_electrons = np.empty_like(incident_photons)
        pes = self.quantum_efficiency(incident_photons)
        for i, (pes, d) in enumerate(zip(pes, self.cfg.pmt_delay.values())):
            anode_electrons[i] = self.tunka_convolve(pes, times, d)
        return anode_electrons

    def gen_FADC_counts(self, pmt_electrons: np.ndarray, g_time_bins: np.ndarray) -> tuple:
        '''This method samples the pmt electrons into NICHE timebins and converts
        those counts into FADC counts.
        '''
        bins = self.NICHE_bins(g_time_bins)
        FADC_count_array = np.empty((len(self.cfg.active_counters), bins.size))
        for i, (pes ,fp) in enumerate(zip(pmt_electrons, self.cfg.fadc_per_pe.values())):
            FADC_count_array[i] = fp * pes[::int(NICHE_TIMEBIN_SIZE)]
        fadc_counts = np.round(FADC_count_array)+ 2048
        fadc_counts[fadc_counts>4096.] = 4096.
        return fadc_counts , bins
    
    @staticmethod
    def rolling_mean_and_var(fadc_counts: np.ndarray) -> np.ndarray:
        '''This method calculates the mean of each trigger window, the mean of 
        each 1024 bin 'waveform', and the variance of each 1024 bin 'waveform'
        '''
        sums = np.cumsum(fadc_counts)
        wfmeans = (sums[WAVEFORM_SIZE:] - sums[:-WAVEFORM_SIZE])/ WAVEFORM_SIZE
        sums_counts2 = np.cumsum(fadc_counts**2)
        wfvars = (sums_counts2[WAVEFORM_SIZE:] - sums_counts2[:-WAVEFORM_SIZE]) / WAVEFORM_SIZE - wfmeans**2
        trig_means = (sums[WAVEFORM_SIZE+TRIGGER_WIDTH:] - sums[WAVEFORM_SIZE:-TRIGGER_WIDTH])/TRIGGER_WIDTH
        return trig_means, wfmeans[:-TRIGGER_WIDTH], wfvars[:-TRIGGER_WIDTH]

    @staticmethod
    def window_means(fadc_counts: np.ndarray) -> np.ndarray:
        '''This method returns the mean of each trigger window from the 
        1024-8th bin.
        '''
        bins_to_avg = fadc_counts[WAVEFORM_SIZE-TRIGGER_WIDTH:]
        bins2d = np.empty((bins_to_avg.size-TRIGGER_WIDTH,TRIGGER_WIDTH))
        for i in range(TRIGGER_WIDTH):
            bins2d[:,i] = bins_to_avg[i:i-TRIGGER_WIDTH]
        return bins2d.mean(axis=1)

    @staticmethod
    def trigger(fadc_counts: np.ndarray) -> np.ndarray:
        '''This method calculates the trigger condition for each bin after the 
        first 1024.
        '''
        trigs = np.full_like(fadc_counts, False, dtype=bool)
        window_means, means, vars = TriggerSim.rolling_mean_and_var(fadc_counts)
        # window_means =Trigger.window_means(fadc_counts)
        trigs[WAVEFORM_SIZE:-TRIGGER_WIDTH] =  (window_means - means)**2  > TRIGGER_VARIANCE * vars
        return trigs

    def gen_triggers(self, fadc_counts: np.ndarray, timebins: np.ndarray) -> tuple[np.ndarray]:
        '''This method returns an array of NICHE 1024 5ns bin snapshots for an
        event and the corresponding times shifted so 0 is the beginning of the window.
        '''
        trigs = np.empty(len(self.cfg.active_counters), dtype = bool)
        snapshots = np.empty(self.array_shape)
        sh_timebins = np.empty(self.array_shape)
        counter = np.arange(fadc_counts.shape[1])
        for i, counts in enumerate(fadc_counts):
            trigger_array = self.trigger(counts)
            trigs[i] = trigger_array.any()
            if trigs[i]:
                trigger_bin = np.argmax(trigger_array)
                snapshots[i] = counts[trigger_bin-500:trigger_bin+524]
                sh_timebins[i] = counter[trigger_bin-500:trigger_bin+524]
            else:
                snapshots[i][:] = -1.
                sh_timebins[i][:] = -1.
        return self.cfg, trigs, snapshots, sh_timebins

@dataclass
class NicheTriggers:
    '''This is the data-container for a niche trigger.
    '''
    cfg: CounterConfig
    trigs: np.ndarray #array of boolean flags for whether each counter triggered
    waveforms: np.ndarray #array of fadc counts for each counter that did trigger
    times: np.ndarray #array of times for each counter that triggered

    def __post_init__(self) -> None:
        self.names = np.array(self.cfg.active_counters)[self.trigs]
        self.waveforms = self.waveforms[self.trigs]
        self.times = self.times[self.trigs]
        self.cts = {}
        for name, wf, t in zip(self.names, self.waveforms, self.times):
            self.cts[name] = CounterTrigger(name,wf,t)

def gen_niche_trigger(ckv: CherenkovOutput) -> NicheTriggers:
    '''This function takes the incident cherenkov light of a shower and simulates the
    niche trigger.
    '''
    incident_ckv_summed = sum_over_wavelengths(ckv.photons)
    incident_photons, photon_bins = gen_photon_signal(incident_ckv_summed,ckv.times)
    # incident_photons += generate_background(ckv.cfg.noise_files)
    photon_times = bin_medians(photon_bins)
    ts = TriggerSim(ckv.cfg)
    electrons = ts.gen_electron_signal(incident_photons, photon_times)
    fadc_counts, NICHE_bins = ts.gen_FADC_counts(electrons, photon_times)
    fadc_counts += generate_background(ckv.cfg.noise_files)
    return NicheTriggers(*ts.gen_triggers(fadc_counts, NICHE_bins))