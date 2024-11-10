import numpy as np
from scipy.signal import chirp
from scipy.io import loadmat
from szkim_library.szkim_fft import perform_fft_comparison
from szkim_library.szkim_filter import process_LFP_data
from szkim_library.szkim_spectrogram import plot_fft_comparison, generate_spectrogram, plot_spectrogram, plot_combined_mean_with_error, plot_tone_spectrograms

# Parameters for chirp signal
duration = 10
f0 = 20
f1 = 500
sampling_rate = 1000
t = np.linspace(0, duration, int(sampling_rate * duration))
chirp_signal = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

# 1) Perform FFT comparison and save plot
frequencies, custom_fft, library_fft = perform_fft_comparison(chirp_signal, sampling_rate)
plot_fft_comparison(frequencies, custom_fft, library_fft, filename="chirp_fft_comparison.png")

# 2) Generate and save spectrogram for the chirp signal
spectrogram_result, frequencies, time_steps = generate_spectrogram(chirp_signal, sampling_rate)
plot_spectrogram(spectrogram_result, frequencies, time_steps, title="Chirp Spectrogram", filename="chirp_spectrogram.png")

# Load LFP data
LFP_data = loadmat('mouseLFP.mat')['DATA']
fs = 10000
cutoff_freq = 1000

# 3) Process LFP data and save combined mean with error bands plot
filtered_tones = process_LFP_data(LFP_data, fs, cutoff_freq)
plot_combined_mean_with_error(filtered_tones['butterworth']['low'], filtered_tones['butterworth']['high'], 'Low-Tone and High-Tone Responses with Error Bands', filename="combined_tone_response.png")

# 4) Save tone spectrograms plot for different filters in a single PNG
plot_tone_spectrograms(filtered_tones, fs, window_size=256, filename="tone_spectrograms.png")
