

from subprocess import PIPE, Popen, run
import numpy as np


def convert_to_PCM(file, original_format='opus', required_format='pcm'):
    cmd = ['ffmpeg', '-i', '-', '-f', 's16le', '-ar', '44100', 'pipe:1' ]
    process = Popen(cmd, stdin=file, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    #print('output', output)
    print('error', error)
    return output


def save_as_wav(pcm_data, out_file='out.wav', output=True, sample_format ='s16le'):
    print(f'saving {out_file} file')
    cmd = ['ffmpeg',
            '-y',                   # override if the file already exists
            '-f', sample_format,          # input format s16le
            #{}"-acodec", "pcm_s16le", # raw pcm data s16 little endian input
            "-acodec", "pcm_"+ sample_format, # raw pcm data s16 little endian input
            '-i', '-',              # pipe input
            '-ac', '1',             # mono
            out_file]              # out file name

    if output:
        print(f'cmd: {cmd}')

    process = run(cmd, input=pcm_data.tobytes(), stdout=PIPE, stderr=PIPE)
    if output:
        print(f'process return code: {process.returncode}')
        print(process.stdout)


def get_PCM(file):
    cmd = ['ffmpeg',
            '-i', file ,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '1',
            'pipe:1' ]
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    #print('output', output)
    print(error)

    #print(output)
    return output


def pcm_samples(file):
    args = ['ffmpeg',
            '-i', file ,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '1',
            'pipe:1' ]
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    #print(error)
    return np.frombuffer(output, dtype='short')

def echo_filter(pcm_samples, input_gain, output_gain, delay, decay):
    echo_arg = "{}:{}:{}:{}".format(input_gain, output_gain, delay, decay)
    print(f'echo_arg is {(echo_arg)}')

    args = ['ffmpeg',
                '-y',                               # Override the existing file if it exists
                '-f', 's16le',                      # input format s16le
                "-acodec", "pcm_s16le",             # raw pcm data s16 little endian input
                '-i', '-',                          # take input file from the pipe,
                '-af',                              # output format
                'aecho=' + echo_arg,                # echo filter argument
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                'pipe:1'
                ]
    print(f'args is {args}')
    process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate(input=pcm_samples.tobytes())
    #print('output =>', output)
    print('error =>', error)
    print(f'length of echo output: {len(output)}')
    return output
