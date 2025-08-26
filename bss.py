import cProfile
import pstats
import sys
from functools import wraps
from pathlib import Path
from time import strftime

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.colors import LinearSegmentedColormap
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
                 Only consider delay yielding estimates in bounds, in samples.
                 (relative `maxd` in the paper)
    n_delay_bins : int, optional
        The range of delay values distributed into bins, default is 50.
        (relative `dbins` in the paper)
    p : int, optional
        Weight the histogram with the symmetric attenuation estimator,
        default is 1.
    q : int, optional
        Weight the histogram with the delay estimator, default is 0.
    assignment_mode : str, optional
        Assignment mode for TF points to sources. One of {'ml', 'nearest', 'radius'}.
        'ml' -> maximum-likelihood (reconstruction error) assignment (default)
        'nearest' -> Euclidean nearest-neighbor in (alpha, delta) space
        'radius' -> keep only TF points within per-axis radii around each peak
        (implemented as an elliptical neighborhood: (Δα/α_r)^2 + (Δδ/δ_r)^2 <= 1)
    alpha_radius : float, optional
        Radius for alpha (attenuation) axis when using 'radius' assignment mode.
        Default is 0.1 * attenuation_max.
    delta_radius : float, optional
        Radius for delta (delay) axis when using 'radius' assignment mode.
        Default is 0.1 * delay_max.
    manual_peaks : tuple, optional
        If provided, use manually specified peaks instead of automatic peak finding.
        Should be a tuple of (δ_peaks, α_peaks) where each is array_like
        of length n_sources. This disables automatic peak finding.
        Note: δ_peaks (delay) should be in range [-delay_max, +delay_max] (in samples)
        and α_peaks (symmetric attenuation) should be in range [-attenuation_max, +attenuation_max].
        The symmetric attenuation α = a - 1/a where a is the relative attenuation factor.

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

        # Plot the input spectrograms (logarithmic frequency axis)
    >>> duet.plot_spectrograms(freq_scale='log')

    # Or use linear frequency axis
    >>> duet.plot_spectrograms(freq_scale='linear')

    # TODO why is this a mirror image spectrogram?

    # Plot the Gabor atoms scatter plot
    >>> duet.plot_atn_delay_scatter(coloring='magnitude')

    # Plot the source classification
    >>> duet.plot_atn_delay_scatter(coloring='classification')

    # Plot sources with magnitude as lightness (best for seeing sources)
    >>> duet.plot_atn_delay_scatter(coloring='magnitude_by_source')

    # Plot colored by (log) frequency
    >>> duet.plot_atn_delay_scatter(coloring='frequency')

    # Plot the attenuation-delay histogram
    >>> duet.plot_atn_delay_hist()

    # Example with manual peaks:
    >>> # Define manual peaks: (δ_peaks, α_peaks) where α = a - 1/a
    >>> manual_delta = np.array([1.5, -0.8, 2.1, -1.2, 0.5])
    >>> manual_alpha = np.array([0.2, -0.1, 0.3, -0.2, 0.1])
    >>> duet_manual = Duet(x, n_sources=5, sample_rate=fs,
    ...                    manual_peaks=(manual_delta, manual_alpha))
    >>> estimates_manual = duet_manual()
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
        assignment_mode='ml',
        alpha_radius=None,
        delta_radius=None,
        manual_peaks=None,
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
        # Assignment of TF points to sources:
        # 'ml'      -> maximum-likelihood (reconstruction error) assignment (default)
        # 'nearest' -> Euclidean nearest-neighbor in (alpha, delta) space
        # 'radius'  -> keep only TF points within per-axis radii around each peak
        #             (implemented as an elliptical neighborhood: (Δα/α_r)^2 + (Δδ/δ_r)^2 <= 1)
        if assignment_mode not in ('ml', 'nearest', 'radius'):
            raise ValueError("assignment_mode must be one of {'ml', 'nearest', 'radius'}")
        self.assignment_mode = assignment_mode

        # Handle manual peaks if provided
        if manual_peaks is not None:
            if not isinstance(manual_peaks, tuple) or len(manual_peaks) != 2:
                raise ValueError("manual_peaks must be a tuple of (δ_peaks, α_peaks)")
            delay_peaks, sym_atn_peaks = manual_peaks

            # Convert to numpy arrays if they aren't already (accepts array_like)
            delay_peaks = np.asarray(delay_peaks)
            sym_atn_peaks = np.asarray(sym_atn_peaks)

            if len(delay_peaks) != n_sources or len(sym_atn_peaks) != n_sources:
                raise ValueError(f"manual_peaks must contain exactly {n_sources} peaks")
            self.manual_peaks = (sym_atn_peaks, delay_peaks)
        else:
            self.manual_peaks = None

        # Default radii as a fraction of configured bounds if not provided
        self.alpha_radius = (
            0.1 * self.attenuation_max if alpha_radius is None else float(alpha_radius)
        )
        self.delta_radius = (
            0.1 * self.delay_max if delta_radius is None else float(delta_radius)
        )

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
        if self.manual_peaks is not None:
            # Use manually specified peaks
            self.sym_atn_peak, self.delay_peak = self.manual_peaks
        else:
            # Use automatic peak finding
            self.sym_atn_peak, self.delay_peak = self._find_n_peaks(
                self.norm_atn_delay_hist, n_peaks=self.n_sources, width=None,
                prominence=0.2)

        # Assign each time-frequency frame to the nearest peak in phase/amplitude
        # space. This partitions the spectrogram into sources (one peak per source)
        self.atn_peak, self.bestind = self._convert_peaks(self.sym_atn_peak)

        # Compute masks for separation
        # (1) Create a binary mask (1 for each tf-point belonging to my source, 0 for others)
        # (2) Mask the spectrogram with the mask created in (1).
        # (3) Rebuild the original wave file from (2).
        est = self._build_masks(self.atn_peak, self.bestind)

        # Check if we detected fewer sources than requested
        if est.shape[0] < self.n_sources:
            raise ValueError(f"Requested {self.n_sources} sources but only {est.shape[0]} were detected. "
                             f"Reduce n_sources to {est.shape[0]} or adjust parameters to detect more sources.")

        return est

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
        # Assume input is already normalized to ±1 floats externally
        self.x1 = self.x[:, mic_pair[0]].astype(np.float64)
        self.x2 = self.x[:, mic_pair[1]].astype(np.float64)

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
        # A = twoDsmooth(A, 3)

        # smooth the histogram - 5x5 gaussian kernel
        gaussian_kernel = np.outer(
            sp.signal.windows.gaussian(5, std=1),
            sp.signal.windows.gaussian(5, std=1)
        )
        gaussian_kernel /= np.sum(gaussian_kernel)
        A = twoDsmooth(A, gaussian_kernel)

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

        if self.assignment_mode == 'ml':
            # Maximum-likelihood (reconstruction error) assignment
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

        # Branch: Nearest-neighbor in (alpha, delta) space
        elif self.assignment_mode == 'nearest':
            # Only consider TF points yielding estimates in bounds; leave others unassigned (0)
            alpha = self.symmetric_atn
            delta = self.delay
            in_bounds_mask = ((np.abs(alpha) < self.attenuation_max) &
                              (np.abs(delta) < self.delay_max))

            # Compute squared Euclidean distance to each peak for all TF points
            # Shapes: peaks -> (n_peaks,), fields -> (F, T)
            # Broadcast to (n_peaks, F, T)
            alpha_diff = alpha[None, ...] - sym_atn_peak[:, None, None]
            delta_diff = delta[None, ...] - self.delay_peak[:, None, None]
            distances_sq = alpha_diff**2 + delta_diff**2

            # Argmin over peaks dimension → indices in [0, n_peaks-1]
            nearest_peak_indices = np.argmin(distances_sq, axis=0)

            # Initialize all as unassigned (0), then fill in-bounds with 1-based indices
            bestind = np.zeros(self.tf1.shape)
            # Create a temporary full map (1-based)
            tmp_full_assignment = nearest_peak_indices + 1
            np.place(bestind, in_bounds_mask, tmp_full_assignment[in_bounds_mask])

            return peaka, bestind

        # Branch: Radius-based assignment in (alpha, delta) with per-axis radii
        elif self.assignment_mode == 'radius':
            # Elliptical neighborhood: keep TF points where (Δα/α_r)^2 + (Δδ/δ_r)^2 <= 1 for any peak.
            # For overlaps, choose the peak with the smallest normalized squared distance.
            # Validate radii
            if not (self.alpha_radius > 0 and self.delta_radius > 0):
                raise ValueError("alpha_radius and delta_radius must be positive for 'radius' assignment_mode")

            alpha = self.symmetric_atn
            delta = self.delay
            in_bounds_mask = ((np.abs(alpha) < self.attenuation_max) &
                            (np.abs(delta) < self.delay_max))

            # Differences to peaks
            alpha_diff = alpha[None, ...] - sym_atn_peak[:, None, None]
            delta_diff = delta[None, ...] - self.delay_peak[:, None, None]

            # Normalized squared distances for membership and tie-breaking; set to inf outside ellipse
            norm_alpha = alpha_diff / self.alpha_radius
            norm_delta = delta_diff / self.delta_radius
            norm_dist_sq = norm_alpha**2 + norm_delta**2
            in_ellipse = norm_dist_sq <= 1.0
            norm_dist_sq = np.where(in_ellipse, norm_dist_sq, np.inf)

            # Choose minimal normalized distance among peaks
            best_peak_indices = np.argmin(norm_dist_sq, axis=0)  # shape: (F, T)
            min_norm_dist = np.take_along_axis(norm_dist_sq, best_peak_indices[None, ...], axis=0)[0]

            # Assign only where inside at least one rectangle and in overall bounds
            assign_mask = np.isfinite(min_norm_dist) & in_bounds_mask
            bestind = np.zeros(self.tf1.shape)
            assignment_1based = best_peak_indices + 1
            np.place(bestind, assign_mask, assignment_1based[assign_mask])

            return peaka, bestind
        else:
            raise ValueError(f"Invalid assignment_mode: {self.assignment_mode}")

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

    def plot_spectrograms(self, freq_scale='log'):
        """
        Plot the spectrograms of the input channels.

        This shows the time-frequency representations that DUET uses
        for source separation.

        Parameters
        ----------
        freq_scale : str, optional
            Frequency axis scale: 'log' or 'linear'.
            Default is 'log'.
        """
        if self.tf1 is None or self.tf2 is None:
            raise RuntimeError("Spectrograms should be computed first "
                               "(call the DUET instance to run the algorithm).")

        # Create time and frequency axes
        time_axis = np.arange(self.tf1.shape[1]) * self._hop_length / self.fs

        # Convert to one-sided spectrum for plotting (positive frequencies only)
        n_pos = self._nfft // 2  # Number of positive frequency bins
        tf1_pos = self.tf1[:n_pos, :]  # Take only positive frequencies
        tf2_pos = self.tf2[:n_pos, :]  # Take only positive frequencies

        # Use rfftfreq for one-sided frequency axis
        freq_axis = np.fft.rfftfreq(self._nfft, 1/self.fs)[1:n_pos+1]  # Skip DC, take positive freqs

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       sharey=True)

        # Normalize to full scale and convert to dBFS
        max_val = max(np.max(np.abs(tf1_pos)),
                      np.max(np.abs(tf2_pos)))
        mag1_db = 20 * np.log10(np.abs(tf1_pos) / max_val + 1e-10)
        mag2_db = 20 * np.log10(np.abs(tf2_pos) / max_val + 1e-10)

        # Plot first channel spectrogram
        im1 = ax1.imshow(mag1_db, aspect='auto', origin='lower',
                         extent=[0, time_axis[-1], freq_axis[0], freq_axis[-1]],
                         cmap='viridis', vmin=-60, vmax=0)
        ax1.set_title(f'Channel 1 Spectrogram (Microphone {self.mic_pair[0]})')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_xlabel('Time (s)')
        if freq_scale == 'log':
            ax1.set_yscale('log')
            ax1.set_ylim(20, freq_axis[-1])
        else:
            ax1.set_ylim(0, freq_axis[-1])
        plt.colorbar(im1, ax=ax1, label='Magnitude (dBFS)')

        # Plot second channel spectrogram
        im2 = ax2.imshow(mag2_db, aspect='auto', origin='lower',
                         extent=[0, time_axis[-1], freq_axis[0], freq_axis[-1]],
                         cmap='viridis', vmin=-60, vmax=0)
        ax2.set_title(f'Channel 2 Spectrogram (Microphone {self.mic_pair[1]})')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        if freq_scale == 'log':
            ax2.set_ylim(20, freq_axis[-1])
        else:
            ax2.set_ylim(0, freq_axis[-1])
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

    def plot_atn_delay_scatter(self, coloring='magnitude', show_peaks=True):
        """
        Plot time-frequency points in attenuation-delay space with different coloring options.

        Parameters
        ----------
        coloring : str, optional
            Coloring scheme: 'magnitude' (dB magnitude), 'classification' (by source),
            'magnitude_by_source' (source colors with magnitude as lightness),
            or 'frequency' (log10 frequency)
        show_peaks : bool, optional
            Whether to show detected peaks as red X markers
        """
        if self.symmetric_atn is None or self.delay is None:
            raise RuntimeError("Attenuation and delay should be computed first (run the algorithm).")

        # Get the attenuation and delay values
        alpha = self.symmetric_atn.flatten()
        delta = self.delay.flatten()

        # Create the scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if coloring == 'magnitude':
            # Get the magnitude values for coloring
            mag1 = np.abs(self.tf1).flatten()
            mag2 = np.abs(self.tf2).flatten()
            magnitude = np.sqrt(mag1 * mag2)
            magnitude_db = 20 * np.log10(magnitude + 1e-10)

            # Filter out low-magnitude points (keep points above -40 dB)
            mask = magnitude_db > -40
            alpha_plot = alpha[mask]
            delta_plot = delta[mask]
            magnitude_db_plot = magnitude_db[mask]

            # Plot with color based on magnitude
            # Delay should be horizontal (x), symmetric attenuation vertical (y)
            scatter = ax.scatter(delta_plot, alpha_plot, c=magnitude_db_plot,
                                 s=2, alpha=0.7, cmap='magma_r',
                                 edgecolors='none', vmin=-40, vmax=0)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Magnitude (dB)')
            ax.set_title('Gabor Atoms in Attenuation-Delay Space')

        elif coloring == 'classification':
            if self.bestind is None:
                raise RuntimeError("Source classification should be computed first (run the algorithm).")

            # Get the classification for each point
            classification = self.bestind.flatten()

            # Filter out unclassified points (value 0)
            mask = classification > 0
            alpha_plot = alpha[mask]
            delta_plot = delta[mask]
            classification_plot = classification[mask]

            # Plot each source with default matplotlib colors
            for i in range(self.n_sources):
                source_mask = classification_plot == (i + 1)
                if np.any(source_mask):
                    ax.scatter(delta_plot[source_mask], alpha_plot[source_mask],
                               s=2, alpha=0.7, label=f'Source {i+1}')

            ax.legend()
            ax.set_title('Time-Frequency Point Classification to Sources')

        elif coloring == 'magnitude_by_source':
            if self.bestind is None:
                raise RuntimeError("Source classification should be computed first (run the algorithm).")

            # Define constants
            ALPHA_MIN = 0.1  # Minimum alpha (most transparent)
            # Minimum magnitude threshold in dB
            MAGNITUDE_THRESHOLD_DB = -20

            # Get magnitude and classification
            mag1 = np.abs(self.tf1).flatten()
            mag2 = np.abs(self.tf2).flatten()
            magnitude = np.sqrt(mag1 * mag2)
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            classification = self.bestind.flatten()

            # Filter out unclassified points and low magnitude
            mask = (classification > 0) & (magnitude_db > MAGNITUDE_THRESHOLD_DB)
            alpha_plot = alpha[mask]
            delta_plot = delta[mask]
            classification_plot = classification[mask]
            magnitude_db_plot = magnitude_db[mask]

            # Plot each source with magnitude as lightness
            scatter_objects = []  # Store scatter objects for colorbar
            for i in range(self.n_sources):
                source_mask = classification_plot == (i + 1)
                if np.any(source_mask):
                    # Normalize magnitude to 0-1 for this source
                    source_mag = magnitude_db_plot[source_mask]
                    if len(source_mag) > 0:
                        mag_norm = (source_mag - source_mag.min()) / (source_mag.max() - source_mag.min() + 1e-10)
                        # Use default color with varying alpha based on magnitude
                        scatter = ax.scatter(delta_plot[source_mask], alpha_plot[source_mask],
                                   s=2, alpha=ALPHA_MIN + (1.0 - ALPHA_MIN)*mag_norm, label=f'Source {i+1}')
                        scatter_objects.append(scatter)

            # Create a colorbar showing the alpha (magnitude) scale
            if scatter_objects:
                # Create a custom colormap that shows transparency levels
                # From transparent to opaque
                colors = [(0, 0, 0, ALPHA_MIN), (0, 0, 0, 1.0)]
                alpha_cmap = LinearSegmentedColormap.from_list('alpha', colors, N=256)

                # Create a dummy scatter custom colorbar showing alpha values
                dummy_scatter = ax.scatter([], [], c=[], cmap=alpha_cmap, vmin=MAGNITUDE_THRESHOLD_DB, vmax=0)
                cbar = plt.colorbar(dummy_scatter, ax=ax)
                cbar.set_label('Magnitude (dBFS) → Alpha')

                # Calculate ticks based on actual magnitude range
                mag_range = 0 - MAGNITUDE_THRESHOLD_DB
                num_ticks = 5
                tick_values = np.linspace(MAGNITUDE_THRESHOLD_DB, 0, num_ticks)
                tick_labels = [f'{tick:.0f} dB' for tick in tick_values]

                cbar.set_ticks(tick_values)
                cbar.set_ticklabels(tick_labels)
                # Remove the dummy scatter
                dummy_scatter.remove()

            ax.legend(loc='upper right')
            ax.set_title('Sources with Magnitude as Lightness')

        elif coloring == 'frequency':
            # Create frequency values for each time-frequency point
            # For two-sided spectrum, we only use positive frequencies for display
            n_pos = self._nfft // 2  # Number of positive frequency bins (excluding DC)
            freq_axis = np.arange(1, n_pos + 1) * self.fs / self._nfft  # 1 to fs/2

            # Create frequency matrix for all TF points (matching the shape of tf1/tf2)
            # tf1/tf2 have shape (n_fft-1, n_time) since DC was removed
            # We need to map frequency indices to actual frequencies
            freq_matrix = np.zeros_like(self.tf1)

            # Fill positive frequency bins (first half)
            freq_matrix[:n_pos, :] = freq_axis[:, None]

            # Fill negative frequency bins (second half) - mirror of positive frequencies
            # For real signals, negative frequencies have same magnitude as positive
            if freq_matrix.shape[0] > n_pos:
                # Negative frequencies: reverse order of positive frequencies (excluding DC and Nyquist)
                neg_freqs = freq_axis[::-1]
                if len(neg_freqs) > freq_matrix.shape[0] - n_pos:
                    neg_freqs = neg_freqs[:freq_matrix.shape[0] - n_pos]
                freq_matrix[n_pos:n_pos+len(neg_freqs), :] = neg_freqs[:, None]

            # Flatten and compute log frequency
            frequency = freq_matrix.flatten().real  # Ensure real values
            log_frequency = np.log10(frequency + 1e-10)  # Add small value to avoid log(0)

            # Filter out low-frequency components (below 20 Hz for better visualization)
            mask = frequency > 20
            alpha_plot = alpha[mask].real  # Ensure real values
            delta_plot = delta[mask].real  # Ensure real values
            log_freq_plot = log_frequency[mask].real  # Ensure real values

            # Define frequency range for colorbar
            max_freq = float(self.fs / 2)  # Ensure real number

            # Plot with color based on log frequency
            scatter = ax.scatter(delta_plot, alpha_plot, c=log_freq_plot,
                                 s=2, alpha=0.7, cmap='magma',
                                 edgecolors='none',
                                 vmin=np.log10(20), vmax=np.log10(max_freq))

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Log₁₀ Frequency (Hz)')

            # Create custom colorbar ticks showing actual frequencies
            freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
            # Filter ticks to valid range
            freq_ticks = [f for f in freq_ticks if f <= max_freq]
            log_freq_ticks = [np.log10(f) for f in freq_ticks]

            # Set colorbar ticks
            cbar.set_ticks(log_freq_ticks)
            cbar.set_ticklabels([f'{f} Hz' for f in freq_ticks])

            ax.set_title('Gabor Atoms Colored by Frequency')

        # Show detected peaks if requested
        if show_peaks and hasattr(self, 'sym_atn_peak') and hasattr(self, 'delay_peak'):
            # Draw a white outline X slightly larger under a black X for contrast
            ax.scatter(self.delay_peak, self.sym_atn_peak,
                       c='white', s=44, marker='x', linewidth=3.0,
                       label='_nolegend_', zorder=5)
            ax.scatter(self.delay_peak, self.sym_atn_peak,
                       c='black', s=36, marker='x', linewidth=1.6,
                       label='Detected Peaks', zorder=6)
            # Label peaks with indices matching output filenames (e.g., output0.wav)
            for i, (d, a) in enumerate(zip(self.delay_peak, self.sym_atn_peak)):
                ax.annotate(
                    f"{i}", (d, a), textcoords="offset points",
                    xytext=(2, -6), ha='left', va='top', color='black', fontsize=9,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground='white')],
                    zorder=7,
                )
            # Ensure legend includes the detected peaks entry
            ax.legend(loc='upper right')

        ax.set_xlabel('Delay (δ)')
        ax.set_ylabel('Symmetric Attenuation (α)')

        # Set axis limits to match histogram bounds (delay on x, attenuation on y)
        ax.set_xlim(-self.delay_max, self.delay_max)
        ax.set_ylim(-self.attenuation_max, self.attenuation_max)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_atn_delay_hist(self):
        """Plot the attenuation-delay histogram as a 3D surface plot."""
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
        plt.xlabel("Delay (δ)", fontsize="xx-large")
        plt.ylabel("Attenuation (α)", fontsize="xx-large")

        plt.tight_layout()
        plt.show()

    def plot_atn_delay_hist_2d(self, log_scale=False):
        """
        Plot the attenuation-delay histogram as a 2D image plot (easier to interpret).

        Parameters
        ----------
        log_scale : bool, optional
            Whether to use logarithmic scaling for better visibility of lower peaks.
            Default is False.
        """
        if self.norm_atn_delay_hist is None:
            raise RuntimeError("A weighted histogram should be computed first.")

        Z = self.norm_atn_delay_hist

        # Apply log scaling if requested
        if log_scale:
            # Add small constant to avoid log(0)
            Z_plot = np.log10(Z + 1e-10)
            vmin = np.min(Z_plot)
            vmax = np.max(Z_plot)
        else:
            Z_plot = Z
            vmin = None
            vmax = None

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # Z has shape (n_attenuation_bins, n_delay_bins): rows -> attenuation (y), cols -> delay (x)
        im = ax.imshow(
            Z_plot,
            extent=[-self.delay_max, self.delay_max,
                    -self.attenuation_max, self.attenuation_max],
            origin='lower',
            aspect='auto',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
        )

        # Mark detected peaks
        if hasattr(self, 'sym_atn_peak') and hasattr(self, 'delay_peak'):
            ax.scatter(self.delay_peak, self.sym_atn_peak,
                       c='red', s=36, marker='x', linewidth=1.5,
                       label='Detected Peaks')
            # Label peaks with indices matching output filenames (e.g., output0.wav)
            for i, (d, a) in enumerate(zip(self.delay_peak, self.sym_atn_peak)):
                ax.annotate(f"{i}", (d, a), textcoords="offset points",
                            xytext=(2, -6), ha='left', va='top',
                            color='red', fontsize=9)
            ax.legend()

        ax.set_xlabel('Delay (δ, samples)')
        ax.set_ylabel('Symmetric Attenuation (α)')
        ax.set_title('Attenuation-Delay Histogram')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if log_scale:
            cbar.set_label('Log10(Weighted Count)')
        else:
            cbar.set_label('Weighted Count')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load separate mono files and combine into stereo
    fs, x1 = sp.io.wavfile.read("Data/x1.wav")
    fs, x2 = sp.io.wavfile.read("Data/x2.wav")
    x1 = x1.astype(np.float64) / np.iinfo(x1.dtype).max
    x2 = x2.astype(np.float64) / np.iinfo(x2.dtype).max
    x = np.column_stack([x1, x2])  # Combine into stereo format (time_steps, 2)
    duet = Duet(x, n_sources=5, sample_rate=fs, attenuation_max=1.5,
                delay_max=2.0,
                assignment_mode="ml",
                #   delta_radius=0.5, alpha_radius=0.5,
                )

    fs, x = sp.io.wavfile.read(r"family reunion screaming kid.wav")
    x = x.astype(np.float64) / np.iinfo(x.dtype).max
    duet = Duet(x[:100000], n_sources=2, sample_rate=fs, attenuation_max=2,
                delay_max=10,  # microphones 7 inches apart = 23 samples
                assignment_mode="ml",
                # assignment_mode="radius", delta_radius=10, alpha_radius=0.3,
                # (δ_peaks, α_peaks)
                manual_peaks=([-1.53061224,  -1], [0.26470588, 0.9]),
                )

    fs, x = sp.io.wavfile.read(r"test_signals/stereo_mix.wav")
    x = x.astype(np.float64) / np.iinfo(x.dtype).max
    duet = Duet(x[:100000], n_sources=5, sample_rate=fs,
                attenuation_max=0.2,
                delay_max=0.8,  # microphones 2 cm apart = 0.9 samples
                assignment_mode="ml",
                # assignment_mode="radius", delta_radius=10, alpha_radius=0.3,
                )


    estimates = duet()

    # Debug: print alpha statistics after DUET processing
    print(f"Alpha statistics:")
    print(f"  Min: {np.min(duet.symmetric_atn):.4f}")
    print(f"  Max: {np.max(duet.symmetric_atn):.4f}")
    print(f"  Mean: {np.mean(duet.symmetric_atn):.4f}")
    print(f"  Std: {np.std(duet.symmetric_atn):.4f}")
    print(f"  Attenuation_max: {duet.attenuation_max}")
    print(f"  Non-zero alpha count: {np.count_nonzero(duet.symmetric_atn)}")
    print(f"  Alpha shape: {duet.symmetric_atn.shape}")
    print(f"  Detected peaks: {duet.sym_atn_peak}")
    for i in range(duet.n_sources):
        sp.io.wavfile.write(f"test_outputs/output{i}.wav", duet.fs,
                            estimates[i, :]+0.05*duet.x1)

    # Plot the input spectrograms
    duet.plot_spectrograms(freq_scale="linear")

    # Plot the Gabor atoms scatter plot
    duet.plot_atn_delay_scatter(coloring='magnitude')

    # Plot the source classification
    duet.plot_atn_delay_scatter(coloring='classification')

    # Plot sources with magnitude as lightness (best for seeing sources)
    duet.plot_atn_delay_scatter(coloring='magnitude_by_source')

    # Plot colored by frequency
    duet.plot_atn_delay_scatter(coloring='frequency')

    # Plot the attenuation-delay histogram
    duet.plot_atn_delay_hist()
    duet.plot_atn_delay_hist_2d(log_scale=False)
