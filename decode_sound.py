'''
    Purpose: Decode the information encoded in an audio file

'''
import numpy as np
import utils
import scipy
from termcolor import colored
import matplotlib.pyplot as plt
import argparse
import encode_sound
import encoders
import decoders
import config



def input_file():
    parser = argparse.ArgumentParser(description='Decode the data encoded in an audio file')
    parser.add_argument('-f', '--file', help="file address")
    args =  parser.parse_args()
    return args.file


if __name__  == '__main__':
    # Read the input file
    file = input_file()
    print(colored(f'input file is {file}', 'yellow'))

    # Read the samples from the input file
    sample_rate, samples = encode_sound.input_audio(file)
    print(f'sample_rate: {sample_rate}  || samples count: {len(samples)}')

    decoders.decoder(samples, sample_rate)
    #decoders.decoder_method2(samples, sample_rate)

    input('Press Enter to return')
