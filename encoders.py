import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import config
import cmath
import joblib
import sys

encoded_places = {}
encoded_places[0] = []
encoded_places[1] = []

def encode_phase_up_value(Zxx, value, start, phi, width, counter):
    global encoded_places

    Yxx = Zxx.copy()
    Yxx = np.transpose(Yxx)             # to iterate over values in a given time

    #rotation_vector = complex(np.cos(phi), np.sin(phi))
    rotation_vector = cmath.rect(1, phi)

    end = start + width
    print(f'{counter}, bit_value: {value} width: {width} location: {start}:{end} ')
    counter = start

    # Encode at even freq bins for bit =1
    # Encode at odd freq bins for bit =0
    if value == 1:
        first = 0
    else:
        first = 1
    for _ in Yxx[start:end]:
        for j in range(config.MIN_FREQ, len(Yxx[0]))[first::2]:
            current_complex =  Yxx[counter][j]
            prev_complex  =  Yxx[counter-1][j]

            amp_factor = np.abs(current_complex)/np.abs(prev_complex)
            Yxx[counter][j] = np.multiply(amp_factor*prev_complex, rotation_vector)

        encoded_places[value].append(counter)
        counter += 1

    return np.transpose(Yxx)


def encode_phase_up(Zxx, data, init_bin=None, phi= np.pi, width= None, margin=None):
    '''
        encode the given data into the STFT 2D array
    '''
    # Use config values for params if not provided
    if not init_bin:
        init_bin = config.INIT_BIN
    if not width:
        width = config.WIDTH_SIZE
    if not margin:
        margin = config.MARGIN_SIZE

    Yxx = Zxx.copy()
    counter = 0
    start = init_bin
    for value in data:
        if start < len(Yxx[0]):
            Yxx = encode_phase_up_value(Yxx, value, start, phi, width, counter)
            start += width + margin
            counter += 1
        else:
            print(colored('SONG IS TOO SMALL TO HOLD DATA','red'))
            print('exiting...')
            sys.exit(0)
            break

    print(f'total bits encoded: {counter}')

    global encoded_places
    joblib.dump(encoded_places, config.ENCODED_PLACES_FILE)
    #print(f'HIGH BIT PLACES ARE', high_bit_places)
    print('1 Encoded at: ', encoded_places[1])
    print('0 Encoded at: ', encoded_places[0])
    return Yxx



def subsequent_phase_jump(Zxx, phi=config.PHASE_JUMP):
    '''Change the phases by pi for every time'''
    Yxx = np.transpose(Zxx.copy())
    rotation_vector = cmath.rect(1, phi)

    for time in range(1,len(Yxx)):
        for freq in range(len(Yxx[0])):
            current_complex = Yxx[time][freq]
            prev_complex = Yxx[time-1][freq]

            amplitude_factor = np.abs(current_complex)/np.abs(prev_complex)
            Yxx[time][freq] = np.multiply(amplitude_factor*prev_complex,
                              rotation_vector)

    return(np.transpose(Yxx))


def multi_channel_phase_modulation(Zxx, data):
    Yxx = np.transpose(Zxx.copy())
    begin = config.INIT_BIN
    total_freq = len(Yxx[0])
    rotation_vector_bit_1 = cmath.rect(1, config.PHASE_JUMP_1)
    rotation_vector_bit_0 = cmath.rect(1, config.PHASE_JUMP_0)

    for byte in data:
        end = begin+config.WIDTH_SIZE
        if end > len(Yxx):
            print(colored('SONG IS TOO SHORT FOR THE GIVEN DATA', 'red'))
            break

        print(colored(f'BYTE: {byte}  begin,end: {begin},{end}', 'green'))
        for time in range(begin,end):
            bit_order = 0
            for bit in byte:
                for freq in range(bit_order, total_freq, config.TOTAL_BITS):

                    current_complex = Yxx[time][freq]
                    prev_complex = Yxx[time-1][freq]
                    amplitude_factor = np.abs(current_complex)/np.abs(prev_complex)
                    # Shift phase by phi_1 if bit == 1
                    if bit == 1:
                        # Yxx[time][freq] = np.multiply(amplitude_factor*prev_complex,
                        #                 rotation_vector_bit_1)
                        Yxx[time][freq] = 2*Yxx[time][freq]
                        #print(time,freq)
                        #print((time,freq), end=', ')
                    # Shift phase by phi_0 if bit == 0
                    # else:
                    #     Yxx[time][freq] = np.multiply(amplitude_factor*prev_complex,
                    #                     rotation_vector_bit_0)
                #print()
                bit_order += 1
        begin = end + config.MARGIN_SIZE


    return np.transpose(Yxx)
