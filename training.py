

import numpy as np
import config
import joblib
import os

def file_is_empty(path):
    return os.stat(path).st_size == 0


def get_high_bit_places(file_name= config.HIGH_BITS_PLACES_FILE):
    return joblib.load(file_name)

def get_encoded_places(filename = config.ENCODED_PLACES_FILE):
    return joblib.load(filename)

def create_training_dataset(arr2D,out_file=config.TRAINING_FILENAME):
    #high_bit_places = get_high_bit_places()
    #print('high_bit_places:', high_bit_places)

    encoded_places = get_encoded_places()
    print('1 encode positions: ', encoded_places[1])
    print('0 encode positions: ', encoded_places[0])

    print(f'out_file is {out_file}')
    arr2D = np.transpose(arr2D.copy())

    print(f'arr2D shape: {arr2D.shape}')

    with open(out_file, 'a+') as out_f:

        # Create header if the file is empty
        if file_is_empty(out_file):
            header = ['x' + str(i) for i in range(len(arr2D[0]))]
            header.append('label')
            out_f.writelines(','.join(header) + '\n')
            print('file is empty. Created headers...')
        else:
            print('file already exists. skipping header creation...')

        for time_bin in range(len(arr2D)):
            line = ''
            for freq_bin in range(len(arr2D[0])):
                value = arr2D[time_bin][freq_bin]
                if np.isnan(value):
                    #print("NAN=>", time_bin, freq_bin)
                    value  = 0
                line  += str(value) + ','

            if time_bin + 1 in encoded_places[1]:
                line += '1'
            elif time_bin + 1 in encoded_places[0]:
                line += '0'
            else:
                line += '2'
            out_f.writelines(line + '\n')



if __name__ == '__main__':
    arr = [[1,2,3,4], [2,3,3,4], [2,3,22,1]]
    create_training_dataset(arr)
