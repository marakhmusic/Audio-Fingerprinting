import numpy as np
import config
from collections import deque

def moving_z_score_technique(series, lag=None, threshold=None, influence=None):
    #https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data

    if not lag:
        lag = config.LAG_MOVING_STD

    if not threshold:
        threshold = config.THRESHOLD_SIGNAL_MOVING_STD

    if not influence:
        influence = config.INFLUENCE_SIGNAL_MOVING_STD

    print(f'lag: {lag},  threshold: {threshold}, influence: {influence}')

    # Define a list which would keep lag number of data points in it
    moving_list = deque(maxlen=lag)

    # Iterate over the given series
    signal = []
    for item in series:
        if len(moving_list) != lag:
            moving_list.append(item)
            signal.append(0)
        else:
            # Calculate standard deviation of the moving list
            std_moving_list = np.std(moving_list)
            mean = np.mean(moving_list)
            # Check if the new item is more than the threshold

            if item > std_moving_list*threshold + mean:
                #print("signal detected: ", item, mean, std_moving_list)
                print(f'signal high: {item:.2f}     {mean:.2f}      {std_moving_list:.2f}')
                # add to the signal
                signal.append(1)
                # add to the moving list with some influence
                # moving_list.append(item*influence)
            else:
                signal.append(0)
                moving_list.append(item)
    return signal
