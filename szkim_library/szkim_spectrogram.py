import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from .szkim_fft import my_fft, next_power_of_2

def generate_spectrogram(signal, sampling_rate):
    """Generate spectrogram using custom FFT and return result."""
    window_size = sampling_rate  # 1 second
    step_size = window_size
    spectrogram_data = []
    for i in range(0, len(signal), step_size):
        windowed_signal = signal[i:i + window_size]
        padded_window = np.pad(windowed_signal, (0, next_power_of_2(len(windowed_signal)) - len(windowed_signal)), 'constant')
        fft_window = my_fft(padded_window)
        spectrogram_data.append(np.abs(fft_window[:len(fft_window) // 2]))
    frequencies = np.fft.fftfreq(next_power_of_2(window_size), 1 / sampling_rate)[:window_size // 2]
    time_steps = np.arange(len(spectrogram_data))
    return np.array(spectrogram_data).T, frequencies, time_steps

# Existing functions

def plot_fft_comparison(frequencies, custom_fft, library_fft, filename="chirp_fft_comparison.png"):
    """Plot and save comparison between custom FFT and numpy FFT."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, np.abs(custom_fft))
    plt.title("Custom FFT Implementation")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(frequencies, np.abs(library_fft))
    plt.title("Library FFT (numpy.fft.fft)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_spectrogram(spectrogram_result, frequencies, time_steps, title="Spectrogram", filename="spectrogram.png"):
    """Plot and save the spectrogram."""
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log(spectrogram_result + 1e-6), aspect='auto', origin='lower', cmap='inferno',
               extent=[time_steps[0], time_steps[-1], frequencies[0], frequencies[-1]])
    plt.colorbar(label='Power (dB)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.savefig(filename)
    plt.show()

def plot_combined_mean_with_error(low_data, high_data, title="Combined Low and High Tone Responses", filename="combined_tone_response.png"):
    """Plot combined mean with error bands for low and high tone responses with custom x and y limits."""
    # Pad data to 3000 ms if necessary
    target_length = 3000
    mean_low = np.mean(low_data, axis=0)
    stderr_low = np.std(low_data, axis=0) / np.sqrt(low_data.shape[0])
    mean_high = np.mean(high_data, axis=0)
    stderr_high = np.std(high_data, axis=0) / np.sqrt(high_data.shape[0])

    # If data length is less than target_length, pad to match target_length
    if len(mean_low) < target_length:
        mean_low = np.pad(mean_low, (0, target_length - len(mean_low)), 'constant', constant_values=np.nan)
        stderr_low = np.pad(stderr_low, (0, target_length - len(stderr_low)), 'constant', constant_values=np.nan)
    if len(mean_high) < target_length:
        mean_high = np.pad(mean_high, (0, target_length - len(mean_high)), 'constant', constant_values=np.nan)
        stderr_high = np.pad(stderr_high, (0, target_length - len(stderr_high)), 'constant', constant_values=np.nan)

    # Plot with updated data lengths
    plt.figure(figsize=(10, 6))
    plt.plot(mean_low, label='Low-Tone Mean', color='blue')
    plt.fill_between(range(len(mean_low)), mean_low - stderr_low, mean_low + stderr_low, color='blue', alpha=0.3)
    plt.plot(mean_high, label='High-Tone Mean', color='orange')
    plt.fill_between(range(len(mean_high)), mean_high - stderr_high, mean_high + stderr_high, color='orange', alpha=0.3)
    
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.xlim(0, target_length)  # Set x-axis limit to 0-3000 ms
    plt.ylim(-25, 10)  # Set y-axis limit to -25 to 10
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_tone_spectrograms(filtered_data, fs, window_size, filename="tone_spectrograms.png"):
    """Plot and save spectrograms for low and high tone responses with individual colorbars."""
    filter_types = list(filtered_data.keys())
    fig, axes = plt.subplots(len(filter_types), 2, figsize=(12, 18), constrained_layout=True)
    freq_limit = 200  # Limit frequency range to 0-200 Hz
    overlap = int(window_size * 0.9)

    for i, filter_type in enumerate(filter_types):
        low_tone_data = np.mean(filtered_data[filter_type]['low'], axis=0)
        high_tone_data = np.mean(filtered_data[filter_type]['high'], axis=0)

        # Apply spectrogram for Low-Tone
        f_low, t_low, Sxx_low = spectrogram(low_tone_data, fs=fs, window='hann', nperseg=window_size, noverlap=overlap)
        freq_idx_low = np.where(f_low <= freq_limit)[0]
        f_filtered_low = f_low[freq_idx_low]
        Sxx_filtered_low = Sxx_low[freq_idx_low, :]

        # Apply spectrogram for High-Tone
        f_high, t_high, Sxx_high = spectrogram(high_tone_data, fs=fs, window='hann', nperseg=window_size, noverlap=overlap)
        freq_idx_high = np.where(f_high <= freq_limit)[0]
        f_filtered_high = f_high[freq_idx_high]
        Sxx_filtered_high = Sxx_high[freq_idx_high, :]

        # Plot Low-Tone Spectrogram
        ax_low = axes[i, 0]
        cax1 = ax_low.pcolormesh(t_low, f_filtered_low, 10 * np.log10(Sxx_filtered_low + 1e-6), shading='gouraud', vmin=-80, vmax=0, cmap='jet')
        ax_low.set_title(f"Low-Tone Spectrogram (Filter: {filter_type})")
        ax_low.set_xlabel("Time (s)")
        ax_low.set_ylabel("Frequency (Hz)")
        ax_low.set_ylim(0, freq_limit)
        fig.colorbar(cax1, ax=ax_low, label="Power (dB)")

        # Plot High-Tone Spectrogram
        ax_high = axes[i, 1]
        cax2 = ax_high.pcolormesh(t_high, f_filtered_high, 10 * np.log10(Sxx_filtered_high + 1e-6), shading='gouraud', vmin=-80, vmax=0, cmap='jet')
        ax_high.set_title(f"High-Tone Spectrogram (Filter: {filter_type})")
        ax_high.set_xlabel("Time (s)")
        ax_high.set_ylabel("Frequency (Hz)")
        ax_high.set_ylim(0, freq_limit)
        fig.colorbar(cax2, ax=ax_high, label="Power (dB)")

    plt.savefig(filename)
    plt.show()
