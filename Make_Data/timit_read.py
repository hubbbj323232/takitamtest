# -*- coding: utf-8 -*-

import numpy as np
import librosa
import glob

def timit_read(timit_path = None, male = None, female = None):
    
    if timit_path is None:
        timit_path = 'TIMIT'

    if male is None:
        male = True

    if female is None:
        female = True



    data_train = []
    data_test  = []
    core_test  = []

    cts_name = 'core_test_set.txt'

    file = open(cts_name, "r")
    for line in file:
        core_test.append(line[0:5])
        

    core_complete_test = 'core'
    

    if (male == True and female == False):
        for filename in glob.glob(timit_path + '/TRAIN/*/M*/*.WAV'):
            data_train.append(filename)
        if (core_complete_test == 'complete'):
            for filename in glob.glob(timit_path + '/TEST/*/M*/*.WAV'):
                data_test.append(filename)
        elif(core_complete_test == 'core'):
            for i in range(len(core_test)):
                for filename in glob.glob(timit_path + '/TEST/*/' + core_test[i] + '/SX*.WAV'):
                    if core_test[i][0] == 'M':
                        data_test.append(filename)
                for filename in glob.glob(timit_path + '/TEST/*/' + core_test[i] + '/SI*.WAV'):
                    if core_test[i][0] == 'M':
                        data_test.append(filename)
        else:
            print("Zla wartosc zmiennej 'core_complete_test'")
            return

    elif (male == False and female == True):
        for filename in glob.glob(timit_path + '/TRAIN/*/F*/*.WAV'):
            data_train.append(filename)
        if (core_complete_test == 'complete'):
            for filename in glob.glob(timit_path + '/TEST/*/F*/*.WAV'):
                data_test.append(filename)
        elif(core_complete_test == 'core'):
            for i in range(len(core_test)):
                for filename in glob.glob(timit_path + '/TEST/*/' + core_test[i] + '/SX*.WAV'):
                    if core_test[i][0] == 'F':
                        data_test.append(filename)
                for filename in glob.glob(timit_path + '/TEST/*/' + core_test[i] + '/SI*.WAV'):
                    if core_test[i][0] == 'F':    
                        data_test.append(filename)
        else:
            print("Zla wartosc zmiennej 'core_complete_test'")
            return

    elif (male == True and female == True):
        for filename in glob.glob(timit_path + '/TRAIN/*/*/*.WAV'):
            data_train.append(filename)
        if (core_complete_test == 'complete'):
            for filename in glob.glob(timit_path + '/TEST/*/*/*.WAV'):
                data_test.append(filename)
        elif(core_complete_test == 'core'):
            for i in range(len(core_test)):
                for filename in glob.glob(timit_path + '/TEST/*/' + core_test[i] + '/SX*.WAV'):
                    data_test.append(filename)
                for filename in glob.glob(timit_path + '/TEST/*/' + core_test[i] + '/SI*.WAV'):
                    data_test.append(filename)
        else:
            print("Zla wartosc zmiennej 'core_complete_test'")
            return
        

    else:
        print ("PUSTA BAZA DANYCH")

    return data_train, data_test




def mixt_rand(data, mixt_number):

    r = np.random.permutation(len(data))
    rand_data = []
    
    for i in range(mixt_number):
        speech = data[r[i]]
        rand_data.append(speech)

    return rand_data





           

           
