import librosa
import numpy as np

def get_features(y, sr, n_mfcc=20, fixed_mfcc_length=20):
    try:
        # Chroma_stft [Short-Time Fourier Transform]
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = chroma_stft.mean()
        chroma_stft_var = chroma_stft.var()

        # RMS [Root Mean Squared]
        rms = librosa.feature.rms(y=y)
        rms_mean = rms.mean()
        rms_var = rms.var()

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = spectral_centroid.mean()
        spectral_centroid_var = spectral_centroid.var()

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = spectral_bandwidth.mean()
        spectral_bandwidth_var = spectral_bandwidth.var()

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = spectral_rolloff.mean()
        spectral_rolloff_var = spectral_rolloff.var()

        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        zero_crossing_rate_mean = zero_crossing_rate.mean()
        zero_crossing_rate_var = zero_crossing_rate.var()

        # Harmony Mean & Perceptual features
        harmonic, perceptual = librosa.effects.hpss(y)
        harmonic_mean = harmonic.mean()
        harmonic_var = harmonic.var()
        perceptual_mean = perceptual.mean()
        perceptual_var = perceptual.var()

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # MFCC (13 MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Ensure all MFCCs have the same length (padding if necessary)
        mfccs_mean = mfccs.mean(axis=1)
        mfccs_var = mfccs.var(axis=1)

        if len(mfccs_mean) < fixed_mfcc_length:
            padding = np.zeros(fixed_mfcc_length - len(mfccs_mean))
            mfccs_mean = np.concatenate((mfccs_mean, padding))

        if len(mfccs_var) < fixed_mfcc_length:
            padding = np.zeros(fixed_mfcc_length - len(mfccs_var))
            mfccs_var = np.concatenate((mfccs_var, padding))

        # Combine features in alternating order
        features = [
            chroma_stft_mean, chroma_stft_var,
            rms_mean, rms_var,
            spectral_centroid_mean, spectral_centroid_var,
            spectral_bandwidth_mean, spectral_bandwidth_var,
            spectral_rolloff_mean, spectral_rolloff_var,
            zero_crossing_rate_mean, zero_crossing_rate_var,
            harmonic_mean, harmonic_var,
            perceptual_mean, perceptual_var,
            tempo[0]
        ]
        
        # Add MFCC features (mean and variance) in alternating order
        for m, v in zip(mfccs_mean, mfccs_var):
            features.append(m)
            features.append(v)

        return np.array(features)

    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}")


if __name__ == "__main__":
    audio = r"YOUR_AUDIO_FILE_PATH.wav"
    y, sr = librosa.load(audio, sr=None)
    result = get_features(y, sr)
    print(result)
