# 🎧 Single-Input Audio Separation (Cocktail Party Problem)

This project demonstrates **single-microphone speech separation** using  
**Non-Negative Matrix Factorization (NMF)** in the time–frequency domain.

With only one microphone, classical ICA cannot be applied.  
Instead, the mixed signal is separated using **spectrogram decomposition and Wiener masking**.

---

## 📌 Features
- Single audio input
- Blind source separation
- Frequency-domain processing
- Soft mask (Wiener) reconstruction
- Clean and readable code

---

## 🧠 Theory Overview
1. Audio is converted to a **Short-Time Fourier Transform (STFT)**
2. Magnitude spectrogram is factorized using **NMF**
3. Soft masks are computed for each source
4. Masks are applied to the original complex spectrogram
5. Inverse STFT reconstructs separated signals

---

## ▶️ Usage

```bash
pip install -r requirements.txt
python separate_audio.py
