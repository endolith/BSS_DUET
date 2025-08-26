#!/usr/bin/env python3
"""
Simple synthetic test signal generator for DUET BSS algorithm.

Creates signals that occupy different time-frequency bins and different
points in the delta-alpha space using pyroomacoustics for realistic
positioning and acoustic effects.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wavfile
import scipy.signal as signal


def generate_chirp(duration, fs, f_start, f_end, t_start=0):
    """Generate a linear chirp signal."""
    t = np.linspace(0, duration, int(duration * fs), False)
    # Create chirp only in specified time window
    chirp = np.zeros_like(t)
    start_idx = int(t_start * fs)
    end_idx = start_idx + int((duration - t_start) * fs)
    if end_idx > len(t):
        end_idx = len(t)

    t_chirp = t[start_idx:end_idx] - t_start
    chirp[start_idx:end_idx] = signal.chirp(
        t_chirp, f_start, t_chirp[-1], f_end, method='linear')
    return chirp


def generate_tone_burst(duration, fs, frequency, t_start, burst_duration, envelope='hann'):
    """Generate a tone burst at specific time and frequency."""
    t = np.linspace(0, duration, int(duration * fs), False)
    tone = np.zeros_like(t)

    start_idx = int(t_start * fs)
    burst_samples = int(burst_duration * fs)
    end_idx = start_idx + burst_samples

    if end_idx > len(t):
        end_idx = len(t)
        burst_samples = end_idx - start_idx

    # Generate tone
    t_tone = np.linspace(0, burst_duration, burst_samples, False)
    tone_signal = np.sin(2 * np.pi * frequency * t_tone)

    # Apply envelope
    if envelope == 'hann':
        window = signal.windows.hann(burst_samples)
        tone_signal *= window

    tone[start_idx:end_idx] = tone_signal
    return tone


def generate_noise_burst(duration, fs, f_low, f_high, t_start, burst_duration):
    """Generate a bandpass filtered noise burst."""
    t = np.linspace(0, duration, int(duration * fs), False)
    noise = np.zeros_like(t)

    start_idx = int(t_start * fs)
    burst_samples = int(burst_duration * fs)
    end_idx = start_idx + burst_samples

    if end_idx > len(t):
        end_idx = len(t)
        burst_samples = end_idx - start_idx

    # Generate white noise
    white_noise = np.random.randn(burst_samples)

    # Bandpass filter
    nyquist = fs / 2
    low = f_low / nyquist
    high = f_high / nyquist
    if high >= 1.0:
        high = 0.99

    b, a = signal.butter(4, [low, high], btype='band')
    filtered_noise = signal.filtfilt(b, a, white_noise)

    # Apply envelope
    window = signal.windows.hann(burst_samples)
    filtered_noise *= window

    noise[start_idx:end_idx] = filtered_noise
    return noise


def create_test_signals(duration=4.0, fs=16000):
    """
    Create a set of test signals that will occupy different regions
    in the time-frequency domain and delta-alpha space.
    """
    # Signal 1: Low frequency chirp (early in time)
    signal1 = generate_chirp(duration, fs, f_start=200, f_end=800, t_start=0.5)

    # Signal 2: High frequency chirp (later in time)
    signal2 = generate_chirp(
        duration, fs, f_start=2000, f_end=4000, t_start=2.0)

    # Signal 3: Pure tone in middle frequency range
    signal3 = generate_tone_burst(
        duration, fs, frequency=1200, t_start=1.0, burst_duration=1.5)

    # Signal 4: Harmonic series (multiple frequencies)
    signal4 = np.zeros(int(duration * fs))
    fundamental = 300
    for harmonic in [1, 2, 3, 4]:
        tone = generate_tone_burst(duration, fs, frequency=fundamental * harmonic,
                                   t_start=0.2, burst_duration=2.0)
        signal4 += tone / harmonic  # Decrease amplitude for higher harmonics

    # Signal 5: Bandpass noise (mid-time)
    signal5 = generate_noise_burst(duration, fs, f_low=1500, f_high=2500,
                                   t_start=1.5, burst_duration=1.0)

    # Normalize all signals to much lower levels to prevent distortion
    signals = [signal1, signal2, signal3, signal4, signal5]
    for i, sig in enumerate(signals):
        if np.max(np.abs(sig)) > 0:
            signals[i] = sig / np.max(np.abs(sig)) * 0.3  # Much lower to prevent clipping

    return signals


def setup_room_and_simulate(signals, fs=16000, anechoic=True):
    """
    Set up room acoustics simulation with pyroomacoustics.
    Creates realistic delays and attenuations based on source positions.

    Parameters:
    -----------
    signals : list
        List of source signals
    fs : int
        Sample rate
    anechoic : bool
        If True, create anechoic environment (no walls/reflections)
        If False, create room with walls and absorption
    """
    # Room dimensions (in meters) - used for positioning regardless of room type
    room_dim = [4, 3, 2.5]  # width, length, height

    if anechoic:
        # Create anechoic environment - no walls, just free field
        room = pra.AnechoicRoom(fs=fs)
        print("Using anechoic environment (no walls/reflections)")
    else:
        # Create room with walls and absorption
        room = pra.ShoeBox(room_dim, fs=fs, absorption=0.2, max_order=3)
        print("Using room with walls and absorption")

    # Microphone positions (stereo pair, 0.02m apart - avoids phase wrapping at 16kHz
    mic_distance = 0.02  # 2cm between mics (about 0.8 inches)
    mic_height = 1.2
    mic_positions = np.array([
        [room_dim[0]/2 - mic_distance/2, room_dim[1]/2, mic_height],  # Left mic
        [room_dim[0]/2 + mic_distance/2, room_dim[1]/2, mic_height]   # Right mic
    ]).T

    room.add_microphone_array(mic_positions)

        # Source positions - arranged in a pentagon around the microphone pair
    # Microphones are at [2.0, 1.5, 1.2] (center of room)
    mic_center = [room_dim[0]/2, room_dim[1]/2, mic_height]

    # Create pentagon with radius ~1.5m around microphone center
    pentagon_radius = 1.5
    pentagon_angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 angles, 0 to 2π

    source_positions = []
    for i, angle in enumerate(pentagon_angles):
        # Position sources in a circle around the microphones
        x = mic_center[0] + pentagon_radius * np.cos(angle)
        y = mic_center[1] + pentagon_radius * np.sin(angle)
        z = mic_center[2]  # Same height as microphones (coplanar)
        source_positions.append([x, y, z])

    # Add sources to room
    for i, (signal_data, pos) in enumerate(zip(signals, source_positions)):
        room.add_source(pos, signal=signal_data)
        print(f"Source {i}: position {pos}")

    # Simulate acoustics
    room.simulate()

    # Get the recorded signals at both microphones
    mic_signals = room.mic_array.signals  # Shape: (n_mics, n_samples)

    return mic_signals, room


def save_signals(signals, mic_signals, fs, output_dir="./"):
    """Save individual source signals and mixed stereo output."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save individual source signals
    for i, signal_data in enumerate(signals):
        filename = f"{output_dir}source_{i:02d}_{get_signal_name(i)}.wav"
        # Convert to int16 for wav file
        signal_int = (signal_data * 32767).astype(np.int16)
        wavfile.write(filename, fs, signal_int)
        print(f"Saved: {filename}")

    # Save stereo mix - normalize to prevent clipping
    stereo_mix = mic_signals.T  # Convert to (n_samples, n_channels)

    # Normalize the mix to prevent clipping
    max_val = np.max(np.abs(stereo_mix))
    if max_val > 0:
        stereo_mix = stereo_mix / max_val * 0.8  # Scale down to 80% of max

    stereo_mix_int = (stereo_mix * 32767).astype(np.int16)
    wavfile.write(f"{output_dir}stereo_mix.wav", fs, stereo_mix_int)
    print(f"Saved: {output_dir}stereo_mix.wav")


def get_signal_name(index):
    """Get descriptive name for each signal type."""
    names = ["Low_Chirp", "High_Chirp", "Pure_Tone", "Harmonic", "BP_Noise"]
    return names[index] if index < len(names) else f"Signal_{index}"


def plot_signals_and_spectrogram(signals, mic_signals, fs):
    """Create visualizations of the generated signals."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    t = np.linspace(0, len(signals[0])/fs, len(signals[0]))

    # Plot individual source signals
    for i, signal_data in enumerate(signals):
        ax = axes[i//2, i % 2] if i < 4 else None
        if ax is not None:
            ax.plot(t, signal_data)
            ax.set_title(f"Source {i}: {get_signal_name(i)}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True)

    # Plot stereo mix spectrogram
    if len(axes) > 2:
        ax_spec = axes[2, 0]
        f, t_spec, Sxx = signal.spectrogram(mic_signals[0], fs, nperseg=1024)
        im = ax_spec.pcolormesh(
            t_spec, f, 10*np.log10(Sxx + 1e-12), shading='gouraud')
        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_xlabel('Time (s)')
        ax_spec.set_title('Left Channel Spectrogram')
        plt.colorbar(im, ax=ax_spec, label='Power (dB)')

        # Plot time domain mix
        ax_time = axes[2, 1]
        t_mix = np.linspace(0, len(mic_signals[0])/fs, len(mic_signals[0]))
        ax_time.plot(t_mix, mic_signals[0], label='Left', alpha=0.7)
        ax_time.plot(t_mix, mic_signals[1], label='Right', alpha=0.7)
        ax_time.set_title('Stereo Mix - Time Domain')
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Amplitude')
        ax_time.legend()
        ax_time.grid(True)

    plt.tight_layout()
    plt.savefig('test_signals_overview.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function to generate test signals."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic test signals for DUET BSS')
    parser.add_argument('--room', action='store_true',
                       help='Enable room acoustics with walls (default: anechoic)')
    parser.add_argument('--duration', type=float, default=4.0,
                       help='Duration in seconds (default: 4.0)')
    parser.add_argument('--fs', type=int, default=16000,
                       help='Sample rate in Hz (default: 16000)')

    args = parser.parse_args()

    print("Generating synthetic test signals for DUET BSS...")

    # Parameters
    duration = args.duration  # seconds
    fs = args.fs  # Hz
    anechoic = not args.room  # Default to anechoic unless --room flag is used

    # Generate individual source signals
    print("\n1. Creating source signals...")
    signals = create_test_signals(duration, fs)

    # Set up room acoustics and simulate
    print("\n2. Setting up room acoustics simulation...")
    mic_signals, room = setup_room_and_simulate(signals, fs, anechoic=anechoic)

    # Save all signals
    print("\n3. Saving signals...")
    save_signals(signals, mic_signals, fs, output_dir="test_signals/")

    # Create visualizations
    print("\n4. Creating visualizations...")
    plot_signals_and_spectrogram(signals, mic_signals, fs)

    print("\nTest signals generated successfully!")
    print("Files saved in test_signals/ directory")
    print("\nTo test with DUET:")
    print("1. Load test_signals/stereo_mix.wav")

    # Print some info about expected delta-alpha distribution
    print(f"\nExpected characteristics:")
    print(f"- {len(signals)} sources arranged in pentagon around microphones")
    print(f"- Microphone separation: 0.02m (2cm, ~0.8 inches)")
    if anechoic:
        print(f"- Environment: Anechoic (free field, no reflections)")
    else:
        print(f"- Environment: Room with walls and absorption")
    print(f"- Room dimensions: 4m x 3m x 2.5m")
    print(f"- Sample rate: {fs} Hz")
    print(f"- Duration: {duration} seconds")


if __name__ == "__main__":
    main()
