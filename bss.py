import cProfile
import pstats
import sys
from functools import wraps
from pathlib import Path
from time import strftime

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.gridspec import GridSpec
from scipy.signal import convolve2d, find_peaks

from find_peaks import find_peak_indices

np.set_printoptions(threshold=sys.maxsize)
FILEDIR = Path(__file__).resolve().parent
EPSILON = 1e-16


def profile_perf(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            result = func(*args, **kwargs)
        with open(f"perf_{strftime(r'%m_%d-%H_%M_%S')}_{func.__name__}.txt",
                  'w', encoding="utf-8") as stream:
            stats = pstats.Stats(pr, stream=stream)
            stats.strip_dirs().sort_stats('tottime').print_stats()
            print(f"function `{func.__name__}` calls in "
                  f"{stats.get_stats_profile().total_tt} seconds")
        return result
    return wrapper


def tfsynthesis(n_sources, timefreqmat, swin, hop_length, n_fft):
    """
    Synthesize the original signals from the time-frequency representation.

    Parameters
    ----------
    n_sources : int
        Number of output channels for the reconstructed signal.
        TODO: This parameter could be derived from timefreqmat.shape[0] and removed.
    timefreqmat : ndarray
        The complex matrix time-freq representation.
        Shape: (n_sources, n_fft, numtime). Note: the original MATLAB version
        expected (n_fft, numtime), but this version handles multiple sources.
    swin : ndarray
        The synthesis window.
    hop_length : int
        The number of samples between adjacent time windows.
    n_fft : int
        The number of frequency components per time point.
        TODO: This parameter is unused and overwritten by timefreqmat.shape[1].
        (equivalent to ``numfreq`` in the original MATLAB version).

    Returns
    -------
    x : ndarray
        The reconstructed signal.
    """
    # MATLAB and Fortran use column-major layout by default,
    # whereas C and C++ use row-major layout
    swin = np.reshape(swin, -1, 'F')

    win_length = swin.size
    _, n_fft, numtime = timefreqmat.shape

    ind = np.fmod(np.arange(win_length), n_fft)
    x = np.zeros((n_sources, (numtime-1) * hop_length + win_length))

    # The original MATLAB version processes only a single channel,
    # but this Python version handles multiple channels simultaneously
    # using broadcasting for ~4x speed improvement.

    # Original code:
    # for i in range(numtime):
    #     temp = n_fft * ifft(timefreqmat[:, i]).real
    #     sind = i * hop_length
    #     for j in range(win_length):
    #         x[sind+j] = x[sind+j] + temp[ind[j]] * swin[j]
    temp = n_fft * np.fft.ifft(timefreqmat, axis=1).real
    for i in range(numtime):
        x[:, i * hop_length: (i+2) * hop_length] += temp[:, ind, i] * swin

    # TODO: Replace all this with scipy.signal.istft?

    return x


def twoDsmooth(mat, ker):
    """
    Smoothing for better identification of the peaks in a graph.

    Could have used Gaussian kernels to do the same but it seemed
    better visual effects were given when this algorithm was followed
    (Again, based on original CASA495).

    Parameters
    ----------
    mat : ndarray
        The 2D matrix to be smoothed.
    ker : int or ndarray
        Either a scalar or a matrix which is used as the averaging kernel.
        If scalar, creates a boxcar kernel of size (ker, ker). For even
        ker values, convolves with additional line kernels to ensure odd size.

    Returns
    -------
    mat : ndarray
        The smoothed matrix.
    """
    try:
        len(ker)
        kmat = ker

    except TypeError:
        kmat = np.ones((ker, ker)) / ker**2

    kr, kc = kmat.shape
    if (kr % 2 == 0):
        kmat = convolve2d(kmat, np.ones((2, 1)), 'same', 'symm')
        kr += 1

    if (kc % 2 == 0):
        kmat = convolve2d(kmat, np.ones((1, 2)), 'same', 'symm')
        kc += 1

    rota = np.rot90(kmat, 2)
    mat = convolve2d(mat, rota, 'same', 'symm')
    return mat


class Duet(object):
    """
    Computes the Degenerate Unmixing Estimation Technique (DUET).

    This class computes the Degenerate Unmixing Estimation Technique
    of an audio signal. It supports a pair of microphones as input.

    Parameters
    ----------
    x : ndarray
        The input audio signal with at least two channels.
        The ndarray must have the following format: (n_channels, time_step).
    n_sources : int
        How many sources want to be separated (maximum observed sources).
        (relative `numsources` in the paper)
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    mic_pair : tuple, optional
        Configure which channel x1 and x2 are, assuming the input is a
        multi-channel audio. It raises an error if input is mono, e.g.
        shape=(1, :) is mono, shape=(2, :) is stereo. The default is
        None (equivalent to tuple(0, 1)).
    attenuation_max : float, optional
        Only consider attenuation yielding estimates in bounds.
        (relative `maxa` in the paper)
    n_attenuation_bins : int, optional
        The range of attenuation values distributed into bins, default is 35.
        (relative `abins` in the paper)
    delay_max : float, optional
        Only consider delay yielding estimates in bounds.
        (relative `maxd` in the paper)
    n_delay_bins : int, optional
        The range of delay values distributed into bins, default is 50.
        (relative `dbins` in the paper)
    p : int, optional
        Weight the histogram with the symmetric attenuation estimator,
        default is 1.
    q : int, optional
        Weight the histogram with the delay estimator, default is 0.

    Examples
    --------
    >>> from bss import Duet
    >>> from scipy.io.wavfile import read, write
    >>> import numpy as np
    >>> # Load separate mono files and combine into stereo
    >>> fs, x1 = read("Data/x1.wav")
    >>> fs, x2 = read("Data/x2.wav")
    >>> x = np.column_stack([x1, x2])  # Combine into stereo format (time_steps, 2)
    >>> duet = Duet(x, n_sources=5, sample_rate=fs)
    >>> estimates = duet()
    >>> for i in range(duet.n_sources):
    >>>     write(f"output{i}.wav", duet.fs, estimates[i, :]+0.05*duet.x1)

    # Plot the input spectrograms
    >>> duet.plot_spectrograms()

    # TODO why is this a mirror image spectrogram?

    # Plot the Gabor atoms scatter plot
    >>> duet.plot_gabor_atoms()

    # Plot the source classification
    >>> duet.plot_source_classification()

    # Plot the attenuation-delay histogram
    >>> duet.plot_atn_delay_hist()
    """

    def __init__(
        self,
        x,
        n_sources,
        sample_rate,
        mic_pair=None,
        attenuation_max=0.7,
        n_attenuation_bins=35,
        delay_max=3.6,
        n_delay_bins=50,
        p=1,
        q=0,
    ):
        self.x = x
        self.n_sources = n_sources
        self.fs = sample_rate
        self.mic_pair = mic_pair
        self.attenuation_max = attenuation_max
        self.n_attenuation_bins = n_attenuation_bins
        self.delay_max = delay_max
        self.n_delay_bins = n_delay_bins
        self.p = p
        self.q = q

        self.x1 = None
        self.x2 = None
        self.tf1 = None
        self.tf2 = None
        self.fmat = None
        self.symmetric_atn = None
        self.delay = None
        self.sym_atn_peak = None
        self.delay_peak = None
        self.atn_peak = None
        self.norm_atn_delay_hist = None
        self.tf_weight = None
        self.bestind = None
        self.prominences = None
        self._nfft = 1024
        self._win_length = 1024
        self._hop_length = 512
        self._awin = np.hamming(1024)

        if self.mic_pair is None:
            self.mic_pair = (0, 1)

    def __call__(self):
        return self.run()

    def run(self):
        # Create the spectrogram of the Left and Right channels, and remove DC
        # component to avoid dividing by zero frequency in the delay estimation.
        self.tf1, self.tf2, self.fmat = self._contruct_histogram(self.mic_pair)

        # For each time/frequency compare the phase and amplitude of the left and
        # right channels. This gives two new coordinates, instead of time-frequency
        # it is phase-amplitude differences.
        self.symmetric_atn, self.delay = self._compute_atn_delay(
            self.tf1, self.tf2, self.fmat)

        # Build a 2-d histogram (one dimension is phase, one is amplitude) where
        # the height at any phase/amplitude is the count of time-frequency bins that
        # have approximately that phase/amplitude.
        self.norm_atn_delay_hist, self.tf_weight = self._compute_weighted_hist(
            self.symmetric_atn, self.delay)

        # Find the location of peaks in the attenuation-delay plane
        self.sym_atn_peak, self.delay_peak = self._find_n_peaks(
            self.norm_atn_delay_hist, n_peaks=self.n_sources, width=0.5,
            prominence=5.0)

        # Assign each time-frequency frame to the nearest peak in phase/amplitude
        # space. This partitions the spectrogram into sources (one peak per source)
        self.atn_peak, self.bestind = self._convert_peaks(self.sym_atn_peak)

        # Compute masks for separation
        # (1) Create a binary mask (1 for each tf-point belonging to my source, 0 for others)
        # (2) Mask the spectrogram with the mask created in (1).
        # (3) Rebuild the original wave file from (2).
        return self._build_masks(self.atn_peak, self.bestind)

    def _contruct_histogram(self, mic_pair):
        """
        Construct the two-dimensional weighted histogram.

        Following the step.1 in the paper.

        Parameters
        ----------
        mic_pair : tuple
            Microphone pair indices.

        Returns
        -------
        tf1 : ndarray
            STFT of x1, the ndarray must have the following format (t, f).
        tf2 : ndarray
            STFT of x2, the ndarray must have the following format (t, f).
        fmat : ndarray
            Frequency matrix, the ndarray must have the following format (t, f).
        """
        # Dividing by maximum to normalize
        self.x1 = self.x[:, mic_pair[0]] / np.iinfo(np.int16).max
        self.x2 = self.x[:, mic_pair[1]] / np.iinfo(np.int16).max

        # time-frequency domain
        _, _, tf1 = sp.signal.stft(self.x1, fs=self.fs, window=self._awin,
                                   nperseg=self._win_length,
                                   return_onesided=False)
        _, _, tf2 = sp.signal.stft(self.x2, fs=self.fs, window=self._awin,
                                   nperseg=self._win_length,
                                   return_onesided=False)

        # removing DC component
        # Since the scipy stft will scale the return value, in order to match the
        # paper result, it should be rescaled back to the origin. Scipy stft pass
        # the scaling == 'spectrum' and mode == 'stft', the values will multiply
        # np.sqrt(1.0 / win.sum()**2). Here are the source codes below.
        # (1) https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/signal/spectral.py#L1174
        # (2) https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/signal/spectral.py#L1806
        tf1 = tf1[1:, :] * self._awin.sum()
        tf2 = tf2[1:, :] * self._awin.sum()

        # calculate positive/negative frequencies for later use in delay calculation
        h1 = np.arange(1, (self._nfft / 2) + 1)
        h2 = np.arange(-(self._nfft / 2) + 1, 0)
        freq = np.concatenate((h1, h2)) * ((2 * np.pi) / self._nfft)

        h = np.ones((tf1.shape[1], freq.shape[0]))
        for i in range(h.shape[0]):
            h[i] *= freq
        fmat = h.transpose()

        return tf1, tf2, fmat

    def _compute_atn_delay(self, tf1, tf2, fmat):
        """
        Calculate the symmetric attenuation (alpha) and delay (delta) for each t-f point.

        Following the step.2 in the paper, 'alpha' relative symmetric attenuation and
        'delta' relative delay.

        Parameters
        ----------
        tf1 : ndarray
            STFT of x1, output from the stft function.
        tf2 : ndarray
            STFT of x2, output from the stft function.
        fmat : ndarray
            Frequency matrix.

        Returns
        -------
        alpha : ndarray
            The symmetric attenuation.
            The ndarray must have the following format (t, f).
        delta : ndarray
            The relative delay.
            The ndarray must have the following format (t, f).
        """
        R21 = (tf2 + EPSILON) / (tf1 + EPSILON)
        a = np.abs(R21)
        alpha = a - 1./a
        delta = -np.imag(np.log(R21)) / fmat

        return alpha, delta

    def _compute_weighted_hist(self, alpha, delta):
        """
        Calculate weighted histogram.

        Following the step.3 in the paper.

        Parameters
        ----------
        alpha : ndarray
            The symmetric attenuation.
            The ndarray must have the following format (t, f).
        delta : ndarray
            The relative delay.
            The ndarray must have the following format (t, f).

        Returns
        -------
        A : ndarray
            A normalized 2D histogram of symmetric attenuation and delay.
            The ndarray must have the following format (alpha, delta).
        tf_weight : ndarray
            Weights. (should add more explanation to this param)
        """
        h1 = np.abs(self.tf1) * np.abs(self.tf2) ** self.p
        h2 = np.abs(self.fmat) ** self.q
        tf_weight = h1 * h2

        # only consider time-frequency points yielding estimates in bounds
        amask = ((np.abs(alpha) < self.attenuation_max) &
                 (np.abs(delta) < self.delay_max))
        alpha_vec = alpha[amask]
        delta_vec = delta[amask]
        tf_weight = tf_weight[amask]

        # determine histogram indices
        alphaind = np.around((self.n_attenuation_bins-1) *
                             (alpha_vec+self.attenuation_max)/(2*self.attenuation_max))
        deltaind = np.around((self.n_delay_bins-1) *
                             (delta_vec+self.delay_max)/(2*self.delay_max))

        # FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
        # A(alphaind(k),deltaind(k)) = tf_weight(k), S is abins-by-dbins
        A = sp.sparse.csr_matrix(
            (tf_weight, (alphaind, deltaind)),
            shape=(self.n_attenuation_bins, self.n_delay_bins)
        ).toarray()

        # smooth the histogram - local average 3-by-3 neighboring bins
        A = twoDsmooth(A, 3)

        return A, tf_weight

    def _find_n_peaks(
        self, norm_atn_delay_hist, n_peaks=None, width=None, threshold=0.2, prominence=None
    ):
        """
        Find the n largest peaks in the 2D histogram.

        Following step 4 in the paper.

        Parameters
        ----------
        norm_atn_delay_hist : ndarray
            A normalized 2D histogram of symmetric attenuation and delay.
            The ndarray must have the following format (alpha, delta).
        n_peaks : int, optional
            How many peaks should be detected. Default is 5.
        width : ndarray, optional
            Required width of peaks in samples.
        threshold : float, optional
            Minimum threshold for peak detection (used by find_peak_indices).
        prominence : number or sequence, optional
            Required prominence of peaks. If None, uses max-peak searching instead.
            Otherwise, passed to scipy.signal.find_peaks. See scipy.signal.find_peaks
            for accepted types.

        Returns
        -------
        atn_peak : ndarray
                    An array containing the peaks of symmetric attenuation.
        The ndarray must have the following format (n_peaks, ).
        delay_peak : ndarray
        An array containing the peaks of delay.
        The ndarray must have the following format (n_peaks, ).
        """
        x = np.linspace(-self.delay_max, self.delay_max, self.n_delay_bins)
        y = np.linspace(-self.attenuation_max, self.attenuation_max,
                        self.n_attenuation_bins)

        if n_peaks is None:
            n_peaks = 5

        if prominence is None:
            print("using max-peak searching")
            # Peaks: [a_idx, d_inx]
            peaks = np.asarray(
                find_peak_indices(norm_atn_delay_hist, n_peaks=n_peaks,
                                  min_dist=1, threshold=threshold))

            cand_peaks = norm_atn_delay_hist[peaks[:, 0], peaks[:, 1]]
            if n_peaks is None:
                std = np.sqrt((np.abs(cand_peaks - cand_peaks[0])**2).mean())
                cand_peaks = cand_peaks[np.abs(cand_peaks - cand_peaks[0]) < std]

            amax_idx = peaks[:cand_peaks.size, 0]
            dmax_idx = peaks[:cand_peaks.size, 1]

        else:
            # https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
            delay_side = np.max(norm_atn_delay_hist, axis=0)

            dmax_idx, prop = find_peaks(
                delay_side,
                width=width,
                prominence=prominence,
            )

            prom_rank = np.argsort(prop['prominences'])[::-1][:n_peaks]
            dmax_idx = dmax_idx[prom_rank]
            self.prominences = prop['prominences'][prom_rank]
            amax_idx = np.argmax(norm_atn_delay_hist[:, dmax_idx], axis=0)

        atn_peak = y[amax_idx]
        delay_peak = x[dmax_idx]

        return atn_peak, delay_peak

    def _convert_peaks(self, sym_atn_peak):
        """
        Determine masks for separation.

        Following the step.5 in the paper.

        Parameters
        ----------
        sym_atn_peak : ndarray
            An array containing the peaks of symmetric attenuation.
            The ndarray must have the following format (n_peaks, ).

        Returns
        -------
        peaka : ndarray
            An array containing the peaks of attenuation.
            The ndarray must have the following format (n_peaks, ).
        bestind : ndarray
            An array containing each source which is a mask.
            The ndarray must have the following format (n_peaks, t, f)
        """
        # convert the symmetric attenuation back to attenuation
        peaka = (sym_atn_peak + np.sqrt(np.square(sym_atn_peak) + 4)) / 2
        bestsofar = float("inf") * np.ones(self.tf1.shape)
        bestind = np.zeros(self.tf1.shape)

        for i in range(sym_atn_peak.size):
            score = (
                np.abs(peaka[i] * np.exp(-1j*self.fmat*self.delay_peak[i]) * self.tf1 - self.tf2) ** 2
            ) / (1 + peaka[i] ** 2)
            mask = score < bestsofar
            s_mask = score[mask]
            np.place(bestind, mask, i+1)
            np.place(bestsofar, mask, s_mask)

        return peaka, bestind

    def _build_masks(self, atn_peak, bestind):
        """
        Demix with ML alignment and convert to time domain.

        Following the step.6 and step.7 in the paper.

        Parameters
        ----------
        atn_peak : ndarray
            An array containing the peaks of attenuation.
            The ndarray must have the following format (n_peaks, ).
        bestind : ndarray
            An array containing each source which is a mask.
            The ndarray must have the following format (n_peaks, t, f).

        Returns
        -------
        est : ndarray
            An array containing a separated wave stream of all speakers.
            The ndarray must have the following format (batch, time_step).
        """
        # 'h' stands for helper, we're using helper variables to break down
        # the logic of what's going on. Apologies for the order of the 'h's
        # Broadcast (a bit faster) the n_sources estimations and return directly.
        # h1     -> (1, 129)
        # h3     -> (1,) * (1023, 129) * (1023, 129)
        # h4     -> (1023, 129) / (1,)
        # h2     -> (1023, 129) * (1023, 129)
        # h      -> (1+1023, 129)
        # new_h1 -> (n_src, 1, 129)
        # new_h3 -> (n_src, None, None) * (n_src, 1023, 129) * (None, 1023, 129)
        # new_h4 -> (None, 1023, 129) / (n_src,)
        # new_h2 -> (n_src, 1023, 129) * (n_src, 1023, 129)
        # new_h  -> (n_src, 1+1023, 129)
        #
        # Original code:
        # est = np.zeros((self.n_sources, self.x1.shape[-1]))
        # for i in range(self.n_sources):
        #     mask = (bestind == i+1)
        #     h1 = np.zeros((1, self.tf1.shape[-1]))
        #     h3 = atn_peak[i] * np.exp(1j*self.fmat*self.delay_peak[i]) * self.tf2
        #     h4 = ((self.tf1+h3) / (1+atn_peak[i]**2))
        #     h2 = h4 * mask
        #     h = np.concatenate((h1, h2))
        #
        #     esti = tfsynthesis(h, np.sqrt(2)*self._awin/1024, self._hop_length, self._nfft)
        #
        #     # add back into the demix a little bit of the mixture
        #     # as that eliminates most of the masking artifacts
        #     est[i] = esti[0:self.x1.shape[-1]]
        #     write(f"out{i}.wav", self.fs, est[i]+0.05*self.x1)
        h3 = (atn_peak[:, None, None]
              * np.exp(1j * self.fmat[None, ...] * self.delay_peak[:, None, None])
              * self.tf2[None, ...])
        h4 = ((self.tf1[None, ...] + h3) / (1 + atn_peak[:, None, None] ** 2))

        # In order to avoid errors caused by the observed source
        # being less than the source we set.
        observed_src = h4.shape[0]
        mask = np.zeros((observed_src, *bestind.shape))
        for i in range(observed_src):
            mask[i, ...] = (bestind == i+1)

        h1 = np.zeros((observed_src, 1, self.tf1.shape[-1]))
        h2 = h4 * mask

        h = np.concatenate((h1, h2), axis=1)

        est = tfsynthesis(observed_src, h, np.sqrt(2)*self._awin/1024,
                          self._hop_length, self._nfft)

        # Normalize each separated source to prevent clipping
        est = est[:, 0:self.x1.shape[-1]]
        for i in range(est.shape[0]):
            if np.max(np.abs(est[i])) > 0:
                est[i] = est[i] / np.max(np.abs(est[i])) * 0.95

        return est

    def plot_spectrograms(self):
        """
        Plot the spectrograms of the input channels.

        This shows the time-frequency representations that DUET uses
        for source separation.
        """
        if self.tf1 is None or self.tf2 is None:
            raise RuntimeError("Spectrograms should be computed first (run the algorithm).")

        # Create time and frequency axes
        time_axis = np.arange(self.tf1.shape[1]) * self._hop_length / self.fs
        freq_axis = np.arange(self.tf1.shape[0]) * self.fs / self._nfft

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       sharey=True)

        # Normalize to full scale and convert to dBFS
        max_val = max(np.max(np.abs(self.tf1)), np.max(np.abs(self.tf2)))
        mag1_db = 20 * np.log10(np.abs(self.tf1) / max_val + 1e-10)
        mag2_db = 20 * np.log10(np.abs(self.tf2) / max_val + 1e-10)

        # Plot first channel spectrogram
        im1 = ax1.imshow(mag1_db, aspect='auto', origin='lower',
                         extent=[0, time_axis[-1], 20, 20000],
                         cmap='viridis', vmin=-60, vmax=0)
        ax1.set_title(f'Channel 1 Spectrogram (Microphone {self.mic_pair[0]})')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_xlabel('Time (s)')
        ax1.set_yscale('log')
        ax1.set_ylim(20, 20000)
        plt.colorbar(im1, ax=ax1, label='Magnitude (dBFS)')

        # Plot second channel spectrogram
        im2 = ax2.imshow(mag2_db, aspect='auto', origin='lower',
                         extent=[0, time_axis[-1], 20, 20000],
                         cmap='viridis', vmin=-60, vmax=0)
        ax2.set_title(f'Channel 2 Spectrogram (Microphone {self.mic_pair[1]})')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_yscale('log')
        ax2.set_ylim(20, 20000)
        plt.colorbar(im2, ax=ax2, label='Magnitude (dBFS)')

        plt.tight_layout()

        # Add interactive hover functionality to show attenuation and delay
        def on_hover(event):
            if event.inaxes in [ax1, ax2]:
                # Get the data coordinates
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    # Convert to array indices
                    time_idx = int(x / time_axis[-1] * self.tf1.shape[1])
                    freq_idx = int(np.log10(y/20) / np.log10(20000/20) * self.tf1.shape[0])

                    # Clamp indices to valid range
                    time_idx = max(0, min(time_idx, self.tf1.shape[1]-1))
                    freq_idx = max(0, min(freq_idx, self.tf1.shape[0]-1))

                    # Get attenuation and delay values
                    if hasattr(self, 'symmetric_atn') and hasattr(self, 'delay'):
                        atn_val = self.symmetric_atn[freq_idx, time_idx]
                        delay_val = self.delay[freq_idx, time_idx]
                        # Use proper signs to maintain consistent title length
                        atn_sign = '+' if atn_val >= 0 else '−'
                        delay_sign = '+' if delay_val >= 0 else '−'
                        ax1.set_title(f'Channel 1 Spectrogram (Microphone {self.mic_pair[0]}) - Attenuation: {atn_sign}{abs(atn_val):.3f}, Delay: {delay_sign}{abs(delay_val):.3f}')
                        ax2.set_title(f'Channel 2 Spectrogram (Microphone {self.mic_pair[1]}) - Attenuation: {atn_sign}{abs(atn_val):.3f}, Delay: {delay_sign}{abs(delay_val):.3f}')
                        plt.draw()

        # Connect the hover event
        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        plt.show()

    def plot_gabor_atoms(self):
        """
        Plot the Gabor atoms (time-frequency points) as a scatter plot in
        attenuation-delay space, colored by magnitude.

        This shows the distribution of time-frequency points and their
        corresponding attenuation-delay values.
        """
        if self.symmetric_atn is None or self.delay is None:
            raise RuntimeError("Attenuation and delay should be computed first (run the algorithm).")

        # Get the attenuation and delay values
        alpha = self.symmetric_atn.flatten()
        delta = self.delay.flatten()

                # Get the magnitude values for coloring
        mag1 = np.abs(self.tf1).flatten()
        mag2 = np.abs(self.tf2).flatten()
        # Use the geometric mean of both channels for coloring
        magnitude = np.sqrt(mag1 * mag2)

        # Convert to dB like the spectrograms
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        # Filter out low-magnitude points (keep points above -40 dB)
        mask = magnitude_db > -40

        # Apply mask to all arrays
        alpha = alpha[mask]
        delta = delta[mask]
        magnitude_db = magnitude_db[mask]

        # Normalize dB magnitude to 0-1 for coloring (from -40 dB to 0 dB)
        magnitude_norm = (magnitude_db + 40) / 40

        # Create the scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot with color based on magnitude
        scatter = ax.scatter(alpha, delta, c=magnitude_db, s=2, alpha=0.7,
                           cmap='viridis', edgecolors='none', vmin=-40, vmax=0)

        ax.set_xlabel('Symmetric Attenuation')
        ax.set_ylabel('Delay')
        ax.set_title('Gabor Atoms in Attenuation-Delay Space')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Magnitude (dB)')

        # Set axis limits to match histogram bounds
        ax.set_xlim(-self.attenuation_max, self.attenuation_max)
        ax.set_ylim(-self.delay_max, self.delay_max)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_source_classification(self):
        """
        Plot the classification of time-frequency points to sources.

        Shows how each time-frequency point is assigned to different sources
        based on their proximity to the detected peaks in attenuation-delay space.
        """
        if self.bestind is None:
            raise RuntimeError("Source classification should be computed first (run the algorithm).")

        # Get the attenuation and delay values
        alpha = self.symmetric_atn.flatten()
        delta = self.delay.flatten()

        # Get the classification for each point
        classification = self.bestind.flatten()

        # Filter out unclassified points (value 0)
        mask = classification > 0
        alpha = alpha[mask]
        delta = delta[mask]
        classification = classification[mask]

                # Create the scatter plot with same axes as gabor atoms
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot each source with default matplotlib colors
        for i in range(self.n_sources):
            source_mask = classification == (i + 1)
            if np.any(source_mask):
                ax.scatter(alpha[source_mask], delta[source_mask],
                          s=2, alpha=0.7, label=f'Source {i+1}')

        # Plot the detected peaks
        if hasattr(self, 'sym_atn_peak') and hasattr(self, 'delay_peak'):
            ax.scatter(self.sym_atn_peak, self.delay_peak,
                      c='red', s=100, marker='x', linewidth=2,
                      label='Detected Peaks')

        ax.set_xlabel('Symmetric Attenuation')
        ax.set_ylabel('Delay')
        ax.set_title('Time-Frequency Point Classification to Sources')
        ax.legend()

        # Set axis limits to match histogram bounds
        ax.set_xlim(-self.attenuation_max, self.attenuation_max)
        ax.set_ylim(-self.delay_max, self.delay_max)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_atn_delay_hist(self):
        if self.norm_atn_delay_hist is None:
            raise RuntimeError("A weighted histogram should be computed first.")

        X = np.linspace(-self.delay_max, self.delay_max, self.n_delay_bins)
        Y = np.linspace(-self.attenuation_max, self.attenuation_max,
                        self.n_attenuation_bins)
        X, Y = np.meshgrid(X, Y)
        Z = self.norm_atn_delay_hist

        fig_hist3d = plt.figure(figsize=(8, 8))
        ax = fig_hist3d.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, alpha=0.8)
        ax.plot(X[0, :], np.max(Z, axis=0), zdir="y", c="hotpink",
                zs=self.attenuation_max)
        ax.plot(Y[:, 0], np.max(Z, axis=1), zdir="x", c="hotpink",
                zs=-self.delay_max)
        ax.contour(X, Y, Z, zdir='z', offset=Z.min()-Z.max())
        ax.set_zlim(Z.min()-Z.max(), Z.max()*1.5)
        ax.tick_params(labelsize="large")
        plt.xlabel("Delay", fontsize="xx-large")
        plt.ylabel("Attenuation", fontsize="xx-large")

        plt.tight_layout()
        plt.show()
