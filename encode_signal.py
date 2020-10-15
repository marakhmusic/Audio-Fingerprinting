import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import signal
import matplotlib
import seaborn
import utils
from termcolor import colored
import scipy.signal
import utils
import subprocess
from matplotlib.colors import ListedColormap
import encoders
import sys
import config
import argparse



def input_file():
    parser = argparse.ArgumentParser(description='Decode the data encoded in an audio file')
    parser.add_argument('-f', '--file', help="file address")
    args =  parser.parse_args()
    return args.file


def on_sigint(signum, frame):
    print('received SIGINT signal, terminating the program now')
    print(f'SIGNUM: {signum} | frame: {frame}')
    sys.exit(0)


def input_audio(file_name = './output/original.wav'):
    '''Handle the input file & return PCM data and sample rate'''
    from scipy.io import wavfile
    sample_rate, samples = wavfile.read(file_name)
    return sample_rate, samples


def phase(values):
    '''Calculate phase for a given value'''
    phases = [ np.arctan(x.imag/x.real) for x in values]
    print(np.around(phases,2))
    return phases

def plot_spectogram():
    # Print the phase spectogram
    overlap = DEFAULT_OVERLAP_RATIO*DEFAULT_WINDOW_SIZE

    Sxx, phases, bins = mlab.specgram(samples, NFFT=DEFAULT_WINDOW_SIZE,
                            Fs= sample_rate, window=mlab.window_hanning,
                            mode='phase', noverlap=overlap)


    print(f'rows {len(Sxx)}, columns: {len(Sxx[0])}')

    # Plot the phase spectogram
    plt.figure('phase spectogram')
    cmap = ListedColormap(['green', 'black', 'red'])
    seaborn.heatmap(Sxx, cmap=cmap)

    plt.show(block=False)

    # Transpose the Spectogram
    Sxx = Sxx.transpose()
    print(f'rows {len(Sxx)}, columns: {len(Sxx[0])}')

    for phases in Sxx:
        print(np.around(phases, 2))



def compute_stft(samples, sample_rate=44100):
    '''Calculate STFT of a given sample'''
    overlap = int(config.DEFAULT_OVERLAP_RATIO*config.DEFAULT_WINDOW_SIZE)
    print(f'overlap: {overlap}')
    print(f'nperseg: {config.DEFAULT_WINDOW_SIZE}')

    freqs, bins, Zxx = scipy.signal.stft(samples, fs=sample_rate,
                        nfft= config.DEFAULT_WINDOW_SIZE,
                        noverlap= overlap,
                        nperseg= config.DEFAULT_WINDOW_SIZE
                        )
    print(f'freqs = {len(freqs)}, bins = {len(bins)}')
    return Zxx, freqs, bins

plt
def plot_heatmap(Zxx, name='plot', block=False, mask=None):
    plt.figure(name)
    #cmap = ListedColormap(['green', 'black', 'red'])

    #seaborn.heatmap(Zxx, cmap=cmap)
    ax = seaborn.heatmap(Zxx, mask=mask, cmap="gray")
    ax.invert_yaxis()

    plt.show(block=block)


def print_samples(Zxx, count):
    # Print samples values for some time bins
    for values in Zxx[:count]:
        print(np.around(values), 2)
        # Print the angle information
        angles = np.angle(values)
        print(colored(np.around(angles), 'yellow') )


def modify_phases(Zxx):

    # modify phases after every 10000 time bins
    Zxx = np.transpose(Zxx)             # Take the transpose
    angles_xx = np.angle(Zxx)
    counter = 0

    for phases in angles_xx:
        # modify if divisible by 10000
        remainder = counter % 10000
        if remainder < 1000:
             angles_xx[counter] += 2*np.pi            # inverse the angle
             angles_xx[counter] = np.arctan(np.tan(angles_xx[counter]))

        counter +=1

    return  np.transpose(angles_xx)                # TODO: return the modified Zxx not the angles_xx


def shift_phase(Zxx, phi=np.pi):
        '''Shift the phase of the given STFT after some interval'''
        # Take the transpose to iterate over phases for each time bin
        Zxx = np.transpose(Zxx.copy())

        # Define the rotation vector that would be multiplied
        rotation_vector = complex(np.cos(phi), np.sin(phi))
        print(f'rotation vector: {rotation_vector}')

        counter = 0
        for values in Zxx:
            remainder = counter % 1000
            if remainder < 100:
                # Rotate every complex number by given phi angle
                Zxx[counter] = np.multiply(Zxx[counter], rotation_vector)

            counter += 1

        # Return the metric in the original shape
        return np.transpose(Zxx)

def fixed_phases(Zxx, phi=np.pi):
    '''Change the phase of the given STFT after some interval to the given one'''
    # Take the transpose to iterate over phases for each time bin
    Zxx = np.transpose(Zxx.copy())

    # Define the rotation vector that would be multiplied
    print(colored(f'fixed phase angle: {phi}'))

    counter = 0
    for values in Zxx:
        counter += 1
        remainder = counter % 10000
        if remainder < 1000:
            # Change the phase to the given angle
            Zxx[counter] = np.abs(Zxx[counter])*complex(np.cos(phi), np.sin(phi))

    # Return the metric in the original shape
    return np.transpose(Zxx)

def get_samples_istft(Zxx, sample_rate):
    '''Return the samples for a given STFT'''
    overlap = config.DEFAULT_OVERLAP_RATIO*config.DEFAULT_WINDOW_SIZE
    _, samples = scipy.signal.istft(Zxx, fs=sample_rate,
                noverlap=overlap, nfft = config.DEFAULT_WINDOW_SIZE )

    return samples.astype('short')


def print_difference(Zxx1, Zxx2):
    Zxx1 = np.transpose(Zxx1)
    Zxx2 = np.transpose(Zxx2)

    counter = 0
    for values1, values2 in zip(Zxx1, Zxx2):
        angles = np.angle(values1) - np.angle(values2)

        if any(angles):
            print([x for x in angles])
            counter += 1
        if counter > 100:
            break





if __name__ == '__main__':
    # Handle keyboard interruption
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    # Read the input file
    audio_file = input_file()
    print(audio_file)
    sample_rate = 44100
    samples = utils.pcm_samples(audio_file)
    print(colored(f'input file sample_rate: {sample_rate}, total samples: {len(samples)} ', 'yellow'))

    # Compute the Short time fourier transform of the samples (STFT)
    Zxx, freqs, bins = compute_stft(samples, sample_rate)
    print(f'size of STFT: {len(Zxx)}, {len(Zxx[0])}')

    # Save the original file again using iSTFT
    samples = get_samples_istft(Zxx, sample_rate)
    utils.save_as_wav2(samples, './phase_modification/original.wav', output=False)


    # plot original phase and frequency spectogram
    plot_heatmap(np.angle(Zxx), name='original phases')

    data = config.DATA
    # Zxx_2 = encoders.encode_method2(Zxx, data, phi=config.PHASE_JUMP)
    # plot_heatmap(np.angle(Zxx_2), name='Zxx_2 angles')
    #
    # # difference in modified and original
    # plot_heatmap(np.angle(Zxx_2) - np.angle(Zxx), name='difference phases')
    #
    # # Save the Zxx_up file again using iSTFT
    # samples = get_samples_istft(Zxx_2, sample_rate)
    # utils.save_as_wav2(samples, './phase_modification/phase_up.wav', output=False)

    Zxx_up = encoders.encode_phase_up(Zxx, data, phi=config.PHASE_JUMP)
    #plot_heatmap(np.angle(Zxx_up), name='Zxx_up angles')

    # difference in modified and original
    plot_heatmap(np.angle(Zxx_up) - np.angle(Zxx), name='difference phases')

    # Save the Zxx_up file again using iSTFT
    samples = get_samples_istft(Zxx_up, sample_rate)
    utils.save_as_wav2(samples, './phase_modification/phase_up.wav', output=False)
    # Zxx_jumped = encoders.jump_phases(Zxx, np.pi)
    # # Save the Zxx_jumped file again using iSTFT
    # samples = get_samples_istft(Zxx_jumped, sample_rate)
    # utils.save_as_wav2(samples, './phase_modification/phase_jumped.wav', output=False)
    #
    # plot_heatmap(np.angle(Zxx_jumped), name='jumped Zxx phases')
    #
    # print('Calculating moving phase difference for jumped ')
    # # Dxx_jumped1 = encoders.moving_phase_difference(Zxx_jumped , 'Dxx_jumped1 avg_diffs')
    # # plot_heatmap(Dxx_jumped1, name='Dxx_jumped1 moving difference')
    #
    # Dxx_jumped2, avg_diffs = encoders.moving_phase_difference2(Zxx_jumped, 'Dxx_jumped2 avg_diffs')
    # plot_heatmap(Dxx_jumped2, name='Dxx_jumped2 moving difference')
    # encoders.decode(avg_diffs, l=10, plot='Dxx_jumped2')

    # print('Calculating moving phase difference ')
    # Dxx = encoders.moving_phase_difference(Zxx)
    # plot_heatmap(Dxx, name='original moving difference')

    # # manipulate the phases by shifting some values by pi
    # modified_Zxx = shift_phase(Zxx, np.pi/2)
    # plot_heatmap(np.angle(modified_Zxx), name='modified phases')
    # print('Calculating moving phase difference ')
    # Dxx = encoders.moving_phase_difference2(modified_Zxx)
    # plot_heatmap(Dxx, name='modified Zxx moving difference')
    #
    #
    # # difference in modified and original
    # plot_heatmap(np.angle(modified_Zxx) - np.angle(Zxx), name='difference phases for phase shifted')
    #
    # # Get audio samples for this modified Zxx
    # samples = get_samples_istft(modified_Zxx, sample_rate)
    # print(f'total samples after istft: {len(samples)}')
    #
    # # Save the samples in a wav file
    # utils.save_as_wav2(samples, out_file='./phase_modification/phase_shifted_file.wav', output=False)


    # # Change the phase to a fixed Value
    # fixed_Zxx = fixed_phases(Zxx, np.pi/2)
    # plot_heatmap(np.angle(fixed_Zxx), name='fixed phases')
    #
    # # difference in modified and original
    # plot_heatmap(np.angle(fixed_Zxx) - np.angle(Zxx), name='difference phases for fixed')
    #
    # # Get audio samples for this modified Zxx
    # samples = get_samples_istft(fixed_Zxx, sample_rate)
    # # Save the samples in a wav file
    # utils.save_as_wav2(samples, out_file='./phase_modification/fixed_phase_file.wav', output=False)



    input('Press Enter to Return')
