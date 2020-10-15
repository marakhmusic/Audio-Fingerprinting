

import utils
import argparse
from argparse import RawTextHelpFormatter
import numpy
import sys
import random
from termcolor import colored



SAMPLE_RATE = 44100

# echo parameters
ECHO_INPUT_GAIN = 0.6
ECHO_OUTPUT_GAIN = 0.3
ECHO_DELAY = 10             # in milli seconds
ECHO_DECAY = 0.1


def get_samples(file):
    # Get the samples
    pcm_buffer = utils.get_PCM(file)
    samples = numpy.frombuffer(pcm_buffer, dtype='short')
    return samples


def delay(samples, time):
    silent_sample_count = int(SAMPLE_RATE*time)
    silent_samples = numpy.ones(silent_sample_count, dtype='short')

    # Add the silence in the beginning to introduce the delay
    samples = numpy.concatenate([silent_samples, samples])
    return samples


def pulsate(samples, pulse_time):

    pulse_len = int(pulse_time*SAMPLE_RATE)
    print(f'total samples: {len(samples)}')
    print(f'pulse time: {pulse_time}')

    chunks_count =  int(len(samples)/pulse_len)
    #print(f'chunks_count: {chunks_count}')

    for chunk in range(chunks_count):
        #print('chunk', chunk)
        if  chunk % 10:
            #print('chunk=>', chunk)
            #print(chunk*pulse_len, pulse_len)
            samples[chunk*pulse_len:chunk*pulse_len + pulse_len] = 0

    return samples


if __name__  == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('-p', '--pulsate', nargs='?')               # pulse size in milliseconds
    parser.add_argument('-d', '--delay', nargs='?')                 # delay in echo
    parser.add_argument('-f', '--file', nargs='?')                  # input file
    args = parser.parse_args()



    print(colored(f'input file: {args.file}', 'yellow'))
    if args.file:
        print(f'the input file is {args.file}')

        # Save the original file
        samples = get_samples(args.file)
        utils.save_as_wav(samples, './output/original.wav')

        # Delay
        #samples = delay(samples, 0.01)
        #utils.save_as_wav(samples, './output/delay.wav')


        # Apply echo filter
        echo_pcm_buffer  = utils.echo_filter(samples, ECHO_INPUT_GAIN,
                        ECHO_OUTPUT_GAIN, ECHO_DELAY, ECHO_DECAY)

        samples = numpy.frombuffer(echo_pcm_buffer, dtype='short')
        samples.setflags(write=True)

        utils.save_as_wav(samples, './output/post_echo.wav')

        # pulsate the samples
        samples = pulsate(samples, 0.01)
        utils.save_as_wav(samples, './output/final_pulsated.wav')


    else:
        print('please provide the input file')
        sys.exit(0)

    #samples = generate_pulses(samples, args.seconds, args.delay)
    #utils.save_as_wav(samples, 'pulsated_song.wav')
