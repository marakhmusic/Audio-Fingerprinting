import numpy as np
import config
import matplotlib.pyplot as plt
import encode_sound
import peak_detection
from collections import deque
import encoders
import seaborn
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from scipy.ndimage.filters import *
from scipy import ndimage
import training
import pickle
from models import create_baseline
import librosa.core
from termcolor import colored



def plot_graph(arr, plot='plot'):
    # Plot arr list
    plt.figure(plot)
    plt.plot(arr)
    plt.show(block=False)


def is_valid_high(arr):
    '''a high is a valid high and not a noise if it is surrounded by other highs (WIDTH_SIZE/2)'''
    high_count = sum(arr)
    if high_count > config.WIDTH_SIZE/2:
        return True
    else:
        return False


def read_data(signal):
    data = []
    width = config.WIDTH_SIZE
    counter = 0
    for i in range(len(signal))[:-width:width]:
        if is_valid_high(signal[i: i+ width]):
            print(f'{counter} found high at, {i}, {i+width}' )
            data.append(1)
            counter +=1
    return data


def plot_signal_by_model(data, total_len):
    signal = np.zeros(total_len)


    # Plot the signal
    plot_graph(signal, 'signal by dl model' )

    # Read the data out of the signal
    data = read_data(signal)
    print("ENCODED DATA IS: ", data)
    print('TOTAL BIT COUNT: ', len(data))


def decode_signal(Zxx):
    Yxx = np.transpose(Zxx.copy())
    total_bits = config.TOTAL_BITS

    # Plot all the bit graph seperately
    # for bit_order in range(total_bits):
    #     Yxx_bitwise = np.transpose(Yxx[:,bit_order::total_bits])
    #     encode_sound.plot_heatmap(Yxx_bitwise, str(bit_order) + ' bit_order')



def print_prediction(prediction, text, end= '\n' ):
    if prediction == 1:
        print(colored(text, 'green'), end=end)
    elif prediction == 0:
        print(colored(text, 'yellow'), end=end)
    if prediction == 2:
        print(colored(text, 'grey'), end=end)


def capture_data(signal):
    data = np.zeros(len(signal))
    for counter in range(len(signal)):
        item = signal[counter]
        if item == 2:
            data[counter] = 0
        elif  item == 0:
            data[counter] =  1
        elif item ==1:
            data[counter] = 2

    plt.figure('captured_data')
    plt.scatter(range(len(data)), data )
    plt.show(block=False)


def get_signal_by_model(Zxx):
    # Load the AI model from local disk
    #pipeline = joblib.load(config.AI_MODEL)
    pipeline = pickle.load(open(config.AI_MODEL, 'rb'))
    # Predict for phase differce line at every time bin
    Yxx = np.transpose(Zxx.copy())
    signal = []
    counter = 0
    for X in Yxx:
        prediction = pipeline.predict([X])[0]
        print_prediction(prediction, counter, end=' ')
        signal.append(prediction)
        counter += 1
    print()

    plot_graph(signal)
    capture_data(signal)

def apply_filter(Zxx):
    Yxx = Zxx.copy()
    Yxx = median_filter(Yxx, size=[100,2])

    #Yxx = ndimage.binary_dilation(Yxx, iterations=1).astype(Yxx.dtype)
    encode_sound.plot_heatmap(Yxx, 'phase_diffs uniform filter')
    return Yxx


def decode(Zxx, step_size=config.DECODE_STEP_SIZE, plot='Decoded Zxx', min_freq=0):
    # Find phase diffs for all frequencies within consecutive time bins
    Yxx = np.transpose(Zxx.copy())
    Yxx = Yxx[min_freq:]

    phase_diffs = np.zeros(Yxx.shape)
    for counter in range(1, len(Yxx)):
        phase_diffs[counter] = np.abs(np.angle(np.divide(Yxx[counter], Yxx[counter-1])))

    # plot phase_diffs
    phase_diffs = np.transpose(phase_diffs)
    encode_sound.plot_heatmap(phase_diffs, 'substracted phases')

    #decode_signal(phase_diffs)

    # # Run average filter
    # Yxx = apply_filter(phase_diffs)

    get_signal_by_model(phase_diffs)

    # Create training data set.
    #training.create_training_dataset(phase_diffs)


def decoder(samples, sample_rate):
    # Compute the Short time fourier transform of the samples (STFT)
    Zxx, freqs, bins = encode_sound.compute_stft(samples, sample_rate)
    print(f'STFT shape: {Zxx.shape}')

    # Plot the heatmap of the phases
    encode_sound.plot_heatmap(np.angle(Zxx), name='Zxx phases')

    print('Calculating moving phase difference for given file')
    decode(Zxx, plot= 'decoded Zxx', min_freq=config.MIN_FREQ)
