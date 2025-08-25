#!/usr/bin/env python3
"""
Simple synthetic signal generator for testing BSS DUET algorithm.

Creates test signals with chirps and tones that occupy different
time-frequency bins and different points in delta-alpha space.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from pathlib import Path


def generate_chirp(t, f0, f1, method='linear'):
    """Generate a chirp signal."""
    return signal.chirp(t, f0, t[-1], f1, method=method)


def generate_tone(t, frequency, amplitude=1.0):
    """Generate a pure tone."""
    return amplitude * np.sin(2 * np.pi * frequency * t)


def apply_delay_attenuation(signal1, signal2, delay_samples, attenuation):
    """
    Apply delay and attenuation to create stereo mix.

    Parameters:
    - signal1, signal2: source signals
    - delay_samples: delay in samples (positive = signal2 delayed)
    - attenuation: attenuation factor for signal2
    """
    # Create stereo channels
    left = signal1 + signal2

    # Apply delay to signal2 in right channel
    if delay_samples > 0:
        delayed_signal2 = np.concatenate([np.zeros(delay_samples), signal2[:-delay_samples]])
    elif delay_samples < 0:
        delayed_signal2 = np.concatenate([signal2[-delay_samples:], np.zeros(-delay_samples)])
    else:
        delayed_signal2 = signal2

    # Apply attenuation and create right channel
    right = signal1 + attenuation * delayed_signal2

    return left, right


def create_test_signal(duration=4.0, sample_rate=16000, save_path="test_signal.wav"):
    """
    Create a synthetic test signal with multiple sources in different TF bins.

    Parameters:
    - duration: signal duration in seconds
    - sample_rate: sampling rate in Hz
    - save_path: output file path
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Initialize combined signal
    total_left = np.zeros_like(t)
    total_right = np.zeros_like(t)

    print(f"Generating test signal: {duration}s at {sample_rate}Hz")

    # Source 1: Low frequency chirp (0-3s, 200-800 Hz)
    # Delta ≈ 5 samples, Alpha ≈ 0.7
    mask1 = (t >= 0) & (t <= 3)
    chirp1 = np.zeros_like(t)
    chirp1[mask1] = 0.6 * generate_chirp(t[mask1], 200, 800)

    left1, right1 = apply_delay_attenuation(
        np.zeros_like(chirp1), chirp1,
        delay_samples=5, attenuation=0.7
    )
    total_left += left1
    total_right += right1
    print("  Added chirp 1: 200-800 Hz, 0-3s, δ=5, α=0.7")

    # Source 2: High frequency chirp (1-4s, 1000-2000 Hz)
    # Delta ≈ -3 samples, Alpha ≈ 1.2
    mask2 = (t >= 1) & (t <= 4)
    chirp2 = np.zeros_like(t)
    chirp2[mask2] = 0.5 * generate_chirp(t[mask2], 1000, 2000)

    left2, right2 = apply_delay_attenuation(
        np.zeros_like(chirp2), chirp2,
        delay_samples=-3, attenuation=1.2
    )
    total_left += left2
    total_right += right2
    print("  Added chirp 2: 1000-2000 Hz, 1-4s, δ=-3, α=1.2")

    # Source 3: Mid frequency tone (0.5-3.5s, 600 Hz)
    # Delta ≈ 8 samples, Alpha ≈ 0.5
    mask3 = (t >= 0.5) & (t <= 3.5)
    tone1 = np.zeros_like(t)
    tone1[mask3] = 0.4 * generate_tone(t[mask3], 600)

    left3, right3 = apply_delay_attenuation(
        np.zeros_like(tone1), tone1,
        delay_samples=8, attenuation=0.5
    )
    total_left += left3
    total_right += right3
    print("  Added tone 1: 600 Hz, 0.5-3.5s, δ=8, α=0.5")

    # Source 4: High frequency tone (2-4s, 1500 Hz)
    # Delta ≈ -6 samples, Alpha ≈ 0.9
    mask4 = (t >= 2) & (t <= 4)
    tone2 = np.zeros_like(t)
    tone2[mask4] = 0.3 * generate_tone(t[mask4], 1500)

    left4, right4 = apply_delay_attenuation(
        np.zeros_like(tone2), tone2,
        delay_samples=-6, attenuation=0.9
    )
    total_left += left4
    total_right += right4
    print("  Added tone 2: 1500 Hz, 2-4s, δ=-6, α=0.9")

    # Normalize to prevent clipping
    max_val = max(np.max(np.abs(total_left)), np.max(np.abs(total_right)))
    if max_val > 0.95:
        total_left = total_left / max_val * 0.95
        total_right = total_right / max_val * 0.95

    # Combine into stereo format (time_samples, 2)
    stereo_signal = np.column_stack([total_left, total_right])

    # Convert to 16-bit integers for WAV format
    stereo_int16 = (stereo_signal * 32767).astype(np.int16)

    # Save as WAV file
    wavfile.write(save_path, sample_rate, stereo_int16)
    print(f"Saved test signal to: {save_path}")

    return stereo_signal, sample_rate


if __name__ == "__main__":
    # Generate the test signal
    signal_data, fs = create_test_signal(
        duration=4.0,
        sample_rate=16000,
        save_path="synthetic_test.wav"
    )

    print(f"\nTest signal properties:")
    print(f"  Shape: {signal_data.shape}")
    print(f"  Duration: {signal_data.shape[0] / fs:.1f} seconds")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Max amplitude: {np.max(np.abs(signal_data)):.3f}")
    print(f"\nThis signal contains 4 overlapping sources with different δ,α parameters:")
    print(f"  - Source 1: δ=5, α=0.7 (chirp 200-800Hz, 0-3s)")
    print(f"  - Source 2: δ=-3, α=1.2 (chirp 1000-2000Hz, 1-4s)")
    print(f"  - Source 3: δ=8, α=0.5 (tone 600Hz, 0.5-3.5s)")
    print(f"  - Source 4: δ=-6, α=0.9 (tone 1500Hz, 2-4s)")
    print(f"\nOverlap periods:")
    print(f"  - 1.0-2.0s: Sources 1,2,3 overlap")
    print(f"  - 2.0-3.0s: All 4 sources overlap")
    print(f"  - 3.0-3.5s: Sources 2,3,4 overlap")
