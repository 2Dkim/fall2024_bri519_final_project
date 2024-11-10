import numpy as np
from scipy.signal import butter, filtfilt, cheby1, cheby2

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def cheby1_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = cheby1(order, 0.05, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def cheby2_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = cheby2(order, 20, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def process_LFP_data(LFP_data, fs, cutoff_freq):
    """Process LFP data with different filters and organize into tones."""
    num_sessions = LFP_data.shape[0]
    num_trials = LFP_data.shape[1]
    filtered_tones = {'butterworth': {'low': [], 'high': []},
                      'cheby1': {'low': [], 'high': []},
                      'cheby2': {'low': [], 'high': []}}

    for session in range(num_sessions):
        for trial in range(num_trials):
            trial_data = LFP_data[session][0][trial]
            low_tone, high_tone = trial_data[:trial_data.shape[0]//2], trial_data[trial_data.shape[0]//2:]

            filtered_tones['butterworth']['low'].append(butter_lowpass_filter(low_tone, cutoff_freq, fs))
            filtered_tones['butterworth']['high'].append(butter_lowpass_filter(high_tone, cutoff_freq, fs))
            filtered_tones['cheby1']['low'].append(cheby1_lowpass_filter(low_tone, cutoff_freq, fs))
            filtered_tones['cheby1']['high'].append(cheby1_lowpass_filter(high_tone, cutoff_freq, fs))
            filtered_tones['cheby2']['low'].append(cheby2_lowpass_filter(low_tone, cutoff_freq, fs))
            filtered_tones['cheby2']['high'].append(cheby2_lowpass_filter(high_tone, cutoff_freq, fs))

    # Convert lists to arrays for each filter type
    for filter_type in filtered_tones:
        filtered_tones[filter_type]['low'] = np.array(filtered_tones[filter_type]['low'])
        filtered_tones[filter_type]['high'] = np.array(filtered_tones[filter_type]['high'])

    return filtered_tones
