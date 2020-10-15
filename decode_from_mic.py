
import sounddevice as sd
import numpy as np
import config
import argparse
from termcolor import colored
import decoders
import utils



def input_duration():
    parser = argparse.ArgumentParser(description='Record and decode data from Sound')
    parser.add_argument('-d', '--duration', help="recording duration", type=int)
    args =  parser.parse_args()
    return args.duration


def record_audio(duration, sample_rate, channels=1):
    total_samples = int(duration*sample_rate)
    samples = sd.rec(total_samples, samplerate=sample_rate,
                     channels=channels,  dtype='short')
    print('Recording started...')

    # Wait for the above recording to complete
    sd.wait()

    #sd.play(samples, sample_rate, blocking=True)
    samples = np.asarray([x[0] for x in samples])
    #print(f'Recording finished. Total samples: {len(samples)}')

    return samples



if __name__ == '__main__':

    duration = input_duration()
    print(colored(f'Recording duration is {duration}', 'yellow'))

    # Record and store samples in a numpy array
    sample_rate = config.SAMPLE_RATE

    samples = record_audio(duration, sample_rate)

    # Save the samples in a wav file
    utils.save_as_wav(samples, './phase_modification/recorded.wav', output=False, sample_format='s16le')

    # Decode the data from samples
    decoders.decoder(samples, sample_rate)

    input('Press Enter to Return')
