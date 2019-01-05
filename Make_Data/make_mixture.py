# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import librosa
from speech_process import speech_process


def make_mixture(data, tr_tst, t_max, noisex_path, noise_type, snr, sample_rate, LC, frameLen, n_fft, hop_length, win_length, nfilt):

    Frame_numb = []
    
    noisex = noisex_path + '/' + noise_type + '.wav'
    noise,sr = librosa.load(noisex, sr = sample_rate)

    mixt_num = len(data)

    for i in range(mixt_num):
        
        speech = data[i]
        x,sr = librosa.load(speech, sr = sample_rate)
        x, index = librosa.effects.trim(x, top_db=40, frame_length=80, hop_length=40)
               
        y,n  = add_noise(x, noise, snr, tr_tst, sr)     

        t_mid = (t_max*sample_rate)/2

        if (x.shape[0] > t_max*sample_rate):

            x = x[len(x)/2 - t_mid : len(x)/2 + t_mid]
            n = n[len(n)/2 - t_mid : len(n)/2 + t_mid]
            y = y[len(y)/2 - t_mid : len(y)/2 + t_mid]
        
        x_mag, x_phase, y_mag, y_phase, ibm, frame_numb, fbank = speech_process(x,y,n, LC, sample_rate, frameLen, n_fft, hop_length, win_length, nfilt)
        
        if (i==0):
            X_mag   = x_mag
            X_phase = x_phase
            Y_mag   = y_mag
            Y_phase = y_phase
            IBM     = ibm
        else:
            X_mag   = np.concatenate((X_mag, x_mag))
            X_phase = np.concatenate((X_phase, x_phase))
            Y_mag   = np.concatenate((Y_mag, y_mag))
            Y_phase = np.concatenate((Y_phase, y_phase))
            IBM     = np.concatenate((IBM, ibm))
        Frame_numb.append(frame_numb)
        
        print("Wykonano {} mieszanin".format(i+1))

    return IBM, Y_mag, Y_phase, X_mag, X_phase, Frame_numb, fbank



def add_noise (x, noise, snr, tr_tst, sample_rate):

    x_len = x.size

    if (tr_tst == 'tr'):
        n = noise[0:120*sample_rate]                #first 2 minutes
    elif (tr_tst == 'tst'):
        n = noise[noise.size-120*sample_rate:]      #last 2 minutes
    else:
        print('Zla wartosc zmiennej tr/tst')
        return
    
       
    q = np.random.randint(n.size - x_len)
    n = n[q:q+x_len]

    rmsx  = np.sqrt(np.mean(x**2))
    rmsn  = np.sqrt(np.mean(n**2))
    ratio = rmsx / rmsn / 10**(snr/20)

    n = ratio*n
    y = x+n
    
    return y, n
