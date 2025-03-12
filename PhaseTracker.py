from math import pi, floor, ceil
from typing import Union, Tuple
from enum import Enum

import scipy
import scipy.signal
import scipy.fft
import numpy as np


class PhaseTrackerMode(Enum):
    fft = 1
    fftwin = 2
    wavelet = 3


class PhaseTracker(object):
    fs = 0.0
    data = np.ndarray(0)

    def __init__(self,
                 fs: float,
                 buffer_len: float = 4.0,
                 mode: PhaseTrackerMode = PhaseTrackerMode.wavelet,
                 analysis_len: float = 2.0,
                 quadrature_len: float = 1.0,
                 freq_limits: Tuple[float, float] = (0.5, 2)):
        self.fs = fs
        self.data_len = int(fs * buffer_len)
        self.data = np.zeros(self.data_len)
        self.mode = mode
        self.analysis_sp = int(fs * analysis_len)
        self.quadrature_sp = int(fs * quadrature_len)  # sp is short for sample
        self.freq_limits = freq_limits

        if mode == PhaseTrackerMode.wavelet:
            # construct the wavelet
            M = int(self.analysis_sp * 2)
            w = 5
            s = lambda f: w * fs / (2 * pi * f)

            # set of frequencies, to identify primary freq
            self.wavelet_freqs = np.linspace(freq_limits[0], freq_limits[1], 30)

            # create wavelet for each frequency, truncated at the middle
            self.wavelet = [scipy.signal.morlet2(M, s(f), w)[:self.analysis_sp] for f in self.wavelet_freqs]

        elif mode == PhaseTrackerMode.fftwin:
            self.fft_window = np.hanning(self.analysis_sp * 2)[:self.analysis_sp]

    def replace_data(self, data: np.ndarray) -> None:
        '''
        Replace the internal buffer with the provided array.

        Parameters
        ----------
        data : np.ndarray
            1D array with new data buffer
        '''
        self.data = data
        self.data_len = data.shape[0]

    def new_data(self, block: np.ndarray) -> None:
        '''
        Roll the internal buffer and append new data block.

        Parameters
        ----------
        block : np.ndarray
            1D array with data to append
            Must be smaller than self.data_len
        '''
        # track append latest block to internal tracking
        n_new_samp = block.size
        self.data[:-1 * n_new_samp] = self.data[n_new_samp:]  # rotate buffer
        self.data[-1 * n_new_samp:] = block

    def estimate(self, block: Union[np.ndarray, None] = None, nfft: int = 4096) -> Tuple[float, float, float, float]:
        '''
        Return estimated phase / amplitude / frequency / quadrature at most recent point of the internal buffer using the wavelet transform.

        If block is provided, this function also rolls the internal buffer and appends the data from block.

        Parameters
        ----------
        block : np.ndarray (Default: None)
            New data

        Returns
        -------
        phase : float
            The current estimated phase

        freq : float
            The current estimated frequency

        amp : float
            The current estimated amplitude

        quadrature : float

        '''
        if block is not None:
            self.new_data(block)

        # the data that we're analyzing
        cdata = self.data[-1 * self.analysis_sp:]

        # apply window if requested
        if self.mode == PhaseTrackerMode.fftwin:
            cdata = cdata * self.fft_window

        if (self.mode == PhaseTrackerMode.fft) or (self.mode == PhaseTrackerMode.fftwin):
            # run FFT on analysis segment
            freqdat = scipy.fft.fft(cdata, n=nfft, workers=-2)

            # identify frequency peak
            freq_limit_idx = np.array(self.freq_limits) / self.fs * nfft
            freq_limit_idx[0] = floor(freq_limit_idx[0])
            freq_limit_idx[1] = ceil(freq_limit_idx[1])

            spectralamp = np.abs(freqdat[freq_limit_idx[0]:freq_limit_idx[1]])  # only data within limits
            max_idx = np.argmax(spectralamp) + freq_limit_idx[0]

            # get phase
            phase_start = np.angle(freqdat[max_idx])
            amp = spectralamp[max_idx] / nfft * 2 * 10

            # get freq
            freq = max_idx / nfft * self.fs

            # estimate sine
            time = np.arange(int(self.analysis_sp / 2)) / self.fs
            phase = ((phase_start + (time * freq * 2 * pi)) % (2 * pi))[-1]

            # compute a forward looking function
            # pred = lambda t: phase[-1] + (t * freq * 2 * np.pi) % (2 * np.pi)

        elif self.mode == PhaseTrackerMode.wavelet:
            # convolve the list of wavelets
            conv_vals = [np.dot(cdata, w) for w in self.wavelet]

            # choose the one with highest amp/phase
            amp_conv_vals = np.abs(conv_vals)
            amp_max = np.argmax(amp_conv_vals)

            # create outputs
            amp = amp_conv_vals[amp_max] / 2
            freq = self.wavelet_freqs[amp_max]
            phase = np.angle(conv_vals[amp_max])

        else:
            raise (NotImplementedError('Unknown mode'))

        ### determine if we're locked on ###
        est_phase = (np.arange(self.quadrature_sp) / self.fs) * freq * 2 * pi
        est_phase = est_phase - est_phase[-1] + phase
        est_sig = np.cos(est_phase)
        est_sig = est_sig / np.trapz(np.abs(est_sig)) * est_sig.size

        # normalize the signal
        normsig = cdata[-self.quadrature_sp:] / np.trapz(np.abs(cdata[-self.quadrature_sp:])) * cdata.size
        quadrature = np.trapz(normsig * est_sig) / cdata.size

        return phase, freq, amp, quadrature