# -*- coding: utf-8 -*-

import numpy as np
import librosa

def speech_process(x,y,n, LC, sample_rate, frameLen, n_fft, hop_length, win_length, nfilt):
 
    X = librosa.core.stft(x, n_fft=n_fft, hop_length = hop_length, win_length = win_length)
    N = librosa.core.stft(n, n_fft=n_fft, hop_length = hop_length, win_length = win_length)
    Y = librosa.core.stft(y, n_fft=n_fft, hop_length = hop_length, win_length = win_length)

    X_mag   = np.absolute(X)
    X_phase = np.angle(X)
    N_mag   = np.absolute(N)
    Y_mag   = np.absolute(Y)
    Y_phase = np.angle(Y)

    fbank = filter_bank(n_fft, nfilt, sample_rate)
    XX = domel(X_mag, fbank)
    NN = domel(N_mag, fbank)
    YY = domel(Y_mag, fbank)
            


    IBM = np.zeros(XX.shape)
    SNR = 20*np.log10(XX / NN)
    IBM = (SNR>LC)*1
    
    X_mag, X_phase, Y_mag, Y_phase, IBM, Frame_numb = convert_data(X_mag, X_phase, YY, Y_phase, IBM, frameLen)

    return X_mag, X_phase, Y_mag, Y_phase, IBM, Frame_numb, fbank
            


def filter_bank (n_fft, nfilt, sample_rate):

    low_freq_mel  = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))        # Convert Hz to Mel
    mel_points    = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)   # Equally spaced in Mel scale
    hz_points     = (700 * (10**(mel_points / 2595) - 1))                 # Convert Mel to Hz
    bin           = np.floor((n_fft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(n_fft / 2 + 1))))

    
    for m in range(1, nfilt + 1):

        f_m_minus = int(bin[m - 1])         # left
        f_m       = int(bin[m])             # center
        f_m_plus  = int(bin[m + 1])         # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    fbank = fbank.T           
    return fbank

def domel(A, fbank):

        AA = fbank.T.dot(A)
        return AA



def convert_data(X_mag, X_phase, Y_mag, Y_phase, IBM, frameLen):

    x_mag, i     = split_data(X_mag, frameLen)
    x_phase, i   = split_data(X_phase, frameLen)
    y_mag, i     = split_data(Y_mag, frameLen)
    y_phase, i   = split_data(Y_phase, frameLen)
    IBM, i       = split_data(IBM, frameLen)

    return x_mag, x_phase, y_mag, y_phase, IBM, i


def split_data(A, frameLen):
    
    fin = False
    i = 0
    
    while (fin != True):
        
        if (i*frameLen + frameLen < A.shape[1]+1):
            i = i+1
        else:
            fin = True
    
    
    lastFrame = frameLen*(i-1) + frameLen
    A_ = A[:,0:lastFrame]
    
    f,t = A_.shape
    
    An = np.zeros((i,f,frameLen))
    
    for k in range(i):
        An[k] = A_[:,frameLen*k : frameLen*k+frameLen]
        
    return An, i


def normalize_train(A):

    A = A + 1e-8
    A = 20 * np.log10(A) 

    u = np.mean(A)
    s = np.std(A)

    An = (A-u)/s

    return An, u, s

def normalize_test(A,u,s):

    A = A + 1e-8
    A = 20 * np.log10(A) 

    An = (A-u)/s

    return An
