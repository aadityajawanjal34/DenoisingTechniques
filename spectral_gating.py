import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from sklearn.metrics import mean_squared_error

# Function to perform Spectral Gating
def spectral_gating(noisy_audio, noise_profile, threshold=1.5, n_fft=2048, hop_length=512):
    """
    Apply Spectral Gating to denoise the audio.
    
    Parameters:
        noisy_audio (np.array): The noisy audio signal.
        noise_profile (np.array): The noise profile (magnitude spectrum of noise).
        threshold (float): The threshold for noise suppression (default: 1.5).
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
    
    Returns:
        denoised_audio (np.array): The denoised audio signal.
    """
    # Compute the STFT of the noisy audio
    f, t, Zxx = stft(noisy_audio, nperseg=n_fft, noverlap=hop_length)
    
    # Compute the magnitude of the STFT
    magnitude = np.abs(Zxx)
    
    # Expand noise_profile to match the shape of magnitude
    noise_profile_expanded = np.expand_dims(noise_profile, axis=1)
    
    # Apply spectral gating: suppress frequencies below the threshold
    threshold_mask = magnitude > (threshold * noise_profile_expanded)
    denoised_magnitude = magnitude * threshold_mask
    
    # Reconstruct the denoised STFT
    denoised_Zxx = denoised_magnitude * np.exp(1j * np.angle(Zxx))
    
    # Compute the inverse STFT to get the denoised audio
    _, denoised_audio = istft(denoised_Zxx, nperseg=n_fft, noverlap=hop_length)
    
    return denoised_audio

# Function to calculate SNR
def calculate_snr(clean_signal, noisy_signal):
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean((clean_signal - noisy_signal)**2)
    return 10 * np.log10(signal_power / noise_power)

# Function to compare MFCCs
def compare_mfcc(clean_audio, denoised_audio, sr, n_mfcc=13):
    # Compute MFCCs for clean and denoised audio
    mfcc_clean = librosa.feature.mfcc(y=clean_audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_denoised = librosa.feature.mfcc(y=denoised_audio, sr=sr, n_mfcc=n_mfcc)
    
    # Ensure both MFCC arrays have the same length along the time axis
    min_length = min(mfcc_clean.shape[1], mfcc_denoised.shape[1])
    mfcc_clean = mfcc_clean[:, :min_length]
    mfcc_denoised = mfcc_denoised[:, :min_length]
    
    # Compute the mean absolute difference
    return np.mean(np.abs(mfcc_clean - mfcc_denoised))

# Function to plot and save combined spectrogram
def plot_and_save_combined_spectrogram(clean_audio, noisy_audio, denoised_audio, sr, filename):
    """
    Plot the combined spectrogram of clean, noisy, and denoised audio and save it as an image file.
    
    Parameters:
        clean_audio (np.array): The clean audio signal.
        noisy_audio (np.array): The noisy audio signal.
        denoised_audio (np.array): The denoised audio signal.
        sr (int): The sample rate.
        filename (str): The filename to save the combined spectrogram image.
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot spectrogram of clean audio
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio)), ref=np.max)
    librosa.display.specshow(D_clean, sr=sr, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set_title("Spectrogram of Clean Audio")
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (Hz)')
    
    # Plot spectrogram of noisy audio
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set_title("Spectrogram of Noisy Audio")
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Frequency (Hz)')
    
    # Plot spectrogram of denoised audio
    D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_audio)), ref=np.max)
    librosa.display.specshow(D_denoised, sr=sr, x_axis='time', y_axis='log', ax=ax[2])
    ax[2].set_title("Spectrogram of Denoised Audio")
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Create the spectral_gating_denoised_auds and spectral_gating_spec folders if they don't exist
if not os.path.exists('spectral_gating_denoised_auds'):
    os.makedirs('spectral_gating_denoised_auds')
if not os.path.exists('spectral_gating_spec'):
    os.makedirs('spectral_gating_spec')

# Initialize lists to store evaluation metrics
mse_values = []
snr_values = []
mfcc_diff_values = []

# Get the list of files in the filtered_auds_augmented folder
files = [f for f in os.listdir('filtered_auds_augmented') if f.endswith('.flac')]
files = files[:100]  # Process only the first 100 files

# Process the first 100 audio files
for file in files:
    # Load clean and noisy audio files
    clean_audio, sr = librosa.load(os.path.join('filtered_auds', file), sr=None)
    noisy_audio, _ = librosa.load(os.path.join('filtered_auds_augmented', file), sr=sr)

    # Estimate noise profile (assuming the first few frames are noise)
    _, _, Zxx_noise = stft(noisy_audio[:2048], nperseg=2048, noverlap=512)
    noise_profile = np.mean(np.abs(Zxx_noise), axis=1)

    # Apply Spectral Gating
    denoised_audio = spectral_gating(noisy_audio, noise_profile, threshold=1.5)

    # Save the denoised audio in the spectral_gating_denoised_auds folder
    output_audio_file = os.path.join('spectral_gating_denoised_auds', file.replace('.flac', '.wav'))
    sf.write(output_audio_file, denoised_audio, sr)

    # Save the combined spectrogram in the spectral_gating_spec folder
    output_spectrogram_file = os.path.join('spectral_gating_spec', file.replace('.flac', '.jpg'))
    plot_and_save_combined_spectrogram(clean_audio, noisy_audio, denoised_audio, sr, output_spectrogram_file)

    # Evaluate using MSE, SNR, and MFCC comparison
    mse = mean_squared_error(clean_audio, denoised_audio[:len(clean_audio)])
    snr = calculate_snr(clean_audio, denoised_audio[:len(clean_audio)])
    mfcc_diff = compare_mfcc(clean_audio, denoised_audio, sr)

    # Store evaluation metrics
    mse_values.append(mse)
    snr_values.append(snr)
    mfcc_diff_values.append(mfcc_diff)

# Calculate min, max, and average of evaluation metrics
mse_min = np.min(mse_values)
mse_max = np.max(mse_values)
mse_avg = np.mean(mse_values)

snr_min = np.min(snr_values)
snr_max = np.max(snr_values)
snr_avg = np.mean(snr_values)

mfcc_diff_min = np.min(mfcc_diff_values)
mfcc_diff_max = np.max(mfcc_diff_values)
mfcc_diff_avg = np.mean(mfcc_diff_values)

# Print the results
print("Evaluation Metrics Summary:")
print(f"MSE - Min: {mse_min}, Max: {mse_max}, Avg: {mse_avg}")
print(f"SNR - Min: {snr_min}, Max: {snr_max}, Avg: {snr_avg}")
print(f"MFCC Difference - Min: {mfcc_diff_min}, Max: {mfcc_diff_max}, Avg: {mfcc_diff_avg}")