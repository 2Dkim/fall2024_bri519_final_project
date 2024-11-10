import numpy as np

def my_fft(x):
    """Custom FFT implementation using recursive approach."""
    N = len(x)
    if N <= 1:
        return x
    even = my_fft(x[0::2])
    odd = my_fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k % len(odd)] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def next_power_of_2(x):
    """Compute the next power of 2 greater than or equal to x."""
    return 1 if x == 0 else 2**(x - 1).bit_length()

def perform_fft_comparison(signal, sampling_rate):
    """Compute custom and library FFT for comparison and return results."""
    padded_signal = np.pad(signal, (0, next_power_of_2(len(signal)) - len(signal)), 'constant')
    custom_fft = my_fft(padded_signal)
    library_fft = np.fft.fft(signal, n=len(padded_signal))
    frequencies = np.fft.fftfreq(len(padded_signal), 1 / sampling_rate)
    return frequencies[:len(frequencies) // 2], custom_fft[:len(custom_fft) // 2], library_fft[:len(library_fft) // 2]
