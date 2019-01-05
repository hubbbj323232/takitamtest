# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy

from timit_read import timit_read, mixt_rand
from make_mixture import make_mixture
from speech_process import normalize_train, normalize_test, filter_bank


###########################   USTAWIENIA    ####################################################

timit_path = 'TIMIT'
noisex_path = 'NOISEX'
male = True
female = True


train_mixt_number = 2000
test_mixt_number  = 192
sample_rate = 8000
#noise_type = 'factory'
#snr = 0.0
t_max = 100              #maksymalna dlugosc nagrania (w sekundach)

n_fft = 512
hop_length = 80
win_length = 200
    
nfilt = 65
frameLen = 5

#make_data  = True
make_spec  = True

###########################   USTAWIENIA   ####################################################



def main(make_data, noise_type, snr):
    
    LC = snr - 5
    
    settings      = 'Dane/settings.npz'
    data0name     = 'Dane/files.npz'
    data1name_tr  = 'Dane/' + noise_type + '_' + str(int(snr)) + '_' + 'tr.npz'
    data1name_tst = 'Dane/' + noise_type + '_' + str(int(snr)) + '_' + 'tst.npz'
    
    print("TWORZENIE DANYCH DLA NAGRAŃ Z SZUMEM {}, snr = {}dB".format(noise_type, snr))

	
    if (make_data == True):
        print("WYBOR NAGRAN")
        data_train, data_test = timit_read(timit_path, male, female)
        data_train = mixt_rand(data_train, train_mixt_number)
        data_test  = mixt_rand(data_test, test_mixt_number)
        np.savez(data0name, data_train = data_train, data_test = data_test)
        print("ZAKONCZONO WYBOR NAGRAN, DANE SA DOSTEPNE W PLIKU {}".format(data0name))
        print(" ")
    else:
        data_train, data_test = load_data(data0name)

    
    
    print("TWORZENIE MIESZANIN")
    print("TRAIN")
    
    nn_target, ytr_mag = make_mixture(data_train, 'tr', t_max, noisex_path, noise_type, snr, sample_rate, LC, frameLen, n_fft, hop_length, win_length, nfilt)[0:2]
    nn_input, u, s = normalize_train(ytr_mag)
    np.savez(data1name_tr, nn_input = nn_input, nn_target = nn_target)
    
    print("TEST")
    nn_target, Y_mag, Y_phase, X_mag, X_phase, frameNumb, fbank = make_mixture(data_test, 'tst', t_max, noisex_path, noise_type, snr, sample_rate, LC, frameLen, n_fft, hop_length, win_length, nfilt)
    nn_input = normalize_test(Y_mag,u,s)
    
    np.savez(data1name_tst, X_mag = X_mag, X_phase = X_phase, Y_mag = Y_mag, Y_phase = Y_phase, nn_input = nn_input, nn_target = nn_target, frameNumb = frameNumb)
    np.savez(settings, train_mixt_number = train_mixt_number, test_mixt_number = test_mixt_number, sample_rate = sample_rate, n_fft = n_fft, hop_length = hop_length, win_length = win_length, nfilt = nfilt, frameLen = frameLen, fbank = fbank)    

    print("ZAKONCZONO TWORZENIE MIESZANIN, DANE SA DOSTEPNE W PLIKACH {} ; {}".format(data1name_tr, data1name_tst))
    print(" ")
               


def load_data(data_name):

    a = np.load(data_name)
    data_train = a['data_train']
    data_test  = a['data_test']

    return data_train, data_test
