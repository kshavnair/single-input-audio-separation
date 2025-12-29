import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import NMF

# Load single mixed audio
y, sr = librosa.load("audio_mixedX1.wav", sr=None, mono=True)

# STFT
n_fft = 2048
hop = 512
Y = librosa.stft(y, n_fft=n_fft, hop_length=hop)
mag = np.abs(Y)

# NMF
nmf = NMF(n_components=2, init="nndsvda", max_iter=2000, random_state=0)
W = nmf.fit_transform(mag)
H = nmf.components_

# Reconstruct magnitudes
S1 = np.outer(W[:, 0], H[0])
S2 = np.outer(W[:, 1], H[1])

# Wiener masks
eps = 1e-8
mask1 = S1 / (S1 + S2 + eps)
mask2 = S2 / (S1 + S2 + eps)

# Apply masks
Y1 = mask1 * Y
Y2 = mask2 * Y

# Inverse STFT
y1 = librosa.istft(Y1, hop_length=hop)
y2 = librosa.istft(Y2, hop_length=hop)

# Normalize
y1 /= np.max(np.abs(y1))
y2 /= np.max(np.abs(y2))

# Save outputs
sf.write("speaker1.wav", y1, sr)
sf.write("speaker2.wav", y2, sr)

print("Separation completed")
