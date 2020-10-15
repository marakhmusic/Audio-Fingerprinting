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
import scipy.io.wavfile
import scipy


def plot_heatmap(Zxx, name='plot', block=False, mask=None):
    #plt.figure(name)
    #cmap = ListedColormap(['green', 'black', 'red'])

    #seaborn.heatmap(Zxx, cmap=cmap)
    #cmap="gray"
    plt.figure(name, figsize=(15, 15)) # width and height in inches

    ax = seaborn.heatmap(Zxx, mask=mask )
    ax.invert_yaxis()

    plt.show(block=block)

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
    print(f'freqbins: {len(freqs)}, time bins: {len(bins)}')
    return Zxx, freqs, bins


def compute_istft(Zxx, sample_rate):
    '''Return the samples for a given STFT'''
    overlap = config.DEFAULT_OVERLAP_RATIO*config.DEFAULT_WINDOW_SIZE

    _, samples = scipy.signal.istft(Zxx, fs=sample_rate,
                noverlap=overlap, nfft = config.DEFAULT_WINDOW_SIZE )

    #return samples.astype('short')
    print('samples_dtype',samples.dtype)
    max = np.max(np.abs(samples))
    print('max: ',max)
    #print([x for x in samples[:1000]])
    return samples/max
    #return samples


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


# def stft(x, fftsize=1024, overlap=4):
#     hop=fftsize//overlap
#     w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]
#     return np.vstack([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])
#
# def istft(X, overlap=4):
#     fftsize=(X.shape[1]-1)*2
#     hop=fftsize//overlap
#     w=scipy.hanning(fftsize+1)[:-1]
#     rcs=int(np.ceil(float(X.shape[0])/float(overlap)))*fftsize
#     print(rcs)
#     x=np.zeros(rcs)
#     wsum=np.zeros(rcs)
#     for n,i in zip(X,range(0,len(X)*hop,hop)):
#         l=len(x[i:i+fftsize])
#         x[i:i+fftsize] += np.fft.irfft(n).real[:l]   # overlap-add
#         wsum[i:i+fftsize] += w[:l]
#     pos = wsum != 0
#     x[pos] /= wsum[pos]
#     return x

def stft(x, fftsize=1024, overlap=4):
    hop = fftsize // overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def istft(X, overlap=4):
    fftsize=(X.shape[1]-1)*2
    hop = fftsize // overlap
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop)
    for n,i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x

if __name__ == '__main__':
    # Handle keyboard interruption
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    # Read the input file
    audio_file = input_file()
    print(audio_file)
    sample_rate = 44100
    sample_rate, samples = input_audio(audio_file)
    print(colored(f'input file sample_rate: {sample_rate}, total samples: {len(samples)} ', 'yellow'))

    # Compute the Short time fourier transform of the samples (STFT)
    Zxx, freqs, bins = compute_stft(samples, sample_rate)

    print(f'Zxx STFT shape: {Zxx.shape}')

    # Save the original file again using iSTFT
    original_samples = compute_istft(Zxx, sample_rate)

    #max_original = np.max(np.abs(original_samples))
    # scipy.io.wavfile.write('./phase_modification/original.wav',
    #                     data=original_samples, rate=sample_rate)
    utils.save_as_wav(original_samples, './phase_modification/original.wav', output=False, sample_format='f64le')

    # plot original phase and frequency spectogram
    plot_heatmap(np.angle(Zxx), name='original phases')

    data = config.DATA

    Zxx_up = encoders.encode_phase_up(Zxx, data, phi=config.PHASE_JUMP)
    plot_heatmap(np.angle(Zxx_up), name='Zxx_up angles')
    # difference in modified and original
    plot_heatmap(np.angle(Zxx_up) - np.angle(Zxx), name='difference phases')

    #Zxx_shifted = encoders.subsequent_phase_jump(Zxx, config.PHASE_JUMP)
    #Zxx_shifted = encoders.multi_channel_phase_modulation(Zxx, data)
    #plot_heatmap(np.angle(Zxx_shifted), name='Zxx_shifted angles')


    # Save the Zxx_up file again using iSTFT
    modified_samples = compute_istft(Zxx_up, sample_rate)
    utils.save_as_wav(modified_samples, './phase_modification/phase_up.wav', output=False, sample_format='f64le')
    #modified_samples = istft(Zxx_shifted)
    # modified_samples = librosa.core.istft(Zxx_shifted,
    #                     hop_length=hop_length,
    #                      center=True)


    #max_modified_samples = np.max(np.abs(modified_samples))
    # scipy.io.wavfile.write('./phase_modification/phase_shifted.wav',
    #                     data=modified_samples/max_modified_samples, rate=sample_rate)

    # Test accuracy
    #decoders.decode(Zxx_shifted)
    Zxx_up_reconstructed, freqs, bins = compute_stft(modified_samples, sample_rate)
    # modified_samples = librosa.core.istft(Zxx_shifted,
    #                     win_length = config.DEFAULT_WINDOW_SIZE,
    #                     hop_length=hop_length,
    #                      center=True)
    # Zxx_modified = librosa.core.stft(modified_samples,
    #                  n_fft=config.DEFAULT_WINDOW_SIZE,
    #                 hop_length = hop_length, center=True)

    # Zxx_constructed = librosa.core.stft(original_samples,
    #                  n_fft=config.DEFAULT_WINDOW_SIZE,
    #                 hop_length = hop_length, center=True)
    plot_heatmap(np.angle(Zxx_up_reconstructed) - np.angle(Zxx),
                name='difference btw Zxx_up_reconstructed  and Zxx')

    # plot_heatmap(np.angle(Zxx_shifted) - np.angle(Zxx_modified),
    #             name='reconstructed modified difference phases')

    #decoders.decode(Zxx_modified)
    input('Press Enter to Return')
