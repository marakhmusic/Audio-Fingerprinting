import numpy as np
import matplotlib.pyplot as plt



def modulator(binary_data):
    modulated = np.multiply(binary_data, 2)
    modulated = modulated - 1
    # modulated = np.substract()
    return modulated


def modulate(binary_signal, L):
    modulated_data = modulator(binary_signal)
    modulated_signal = np.repeat(modulated_data, L)
    return modulated_signal

if __name__ == '__main__':
    L = 50      # Oversampling factor
    signal = [0, 1, 0, 1, 1, 0, 1]

    total_symbols =  len(signal)
    x = np.arange(0,total_symbols, 1/L)

    carrier_frequency = 5              # Carrier frequency
    carrier_amplitude= 10             # Carrier amplitude
    noise_frequency = 2
    noise_amplitude  = 1

    carrier = carrier_amplitude*np.sin(2*np.pi*carrier_frequency*x)

    modulated_signal = modulate(signal, L)

    #print('time_axis', time_axis)
    plt.plot(x, modulated_signal)

    plt.figure('carrier wave')
    plt.plot(x, carrier)

    #transmission =   carrier*modulated_signal
    transmission =carrier_amplitude*np.sin(2*np.pi*carrier_frequency*x + modulated_signal)

    white_noise = noise_amplitude*np.sin(2*np.pi*noise_frequency*x)
    plt.figure('transmission')
    plt.plot(x, transmission)

    recieved = transmission + white_noise
    plt.figure('recieved')
    plt.plot(x, recieved)

    plt.show(block=False)
    input('Enter to continue')
