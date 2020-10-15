import numpy as np


# Keep all the global variables here
SAMPLE_RATE = 44100


# count of time bins
MARGIN_SIZE = 20        # left and right margin for a high
WIDTH_SIZE = 10    # padding /width of the high

INIT_BIN = 20
DECODE_STEP_SIZE  = 1


DEFAULT_WINDOW_SIZE = 1024
DEFAULT_OVERLAP_RATIO = 0.5

#MIN_FREQ = DEFAULT_WINDOW_SIZE//4
MIN_FREQ = 0

PHASE_JUMP = np.pi

# DATA = [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0 , 1, 0, 1, 0,
#         1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0 , 1, 0, 1, 0,
#         1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0 , 1, 0, 1, 0,
#         1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0 , 1, 0, 1, 0]

DATA = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        
#DATA = [1 for x in range(3000)]         # 1500 one !!
# DATA=[[1,0,0,0,0,0,0,0],
#       [1,1,1,1,1,1,1,1],
#       [0,1,0,0,0,0,0,0],
#       [0,0,1,0,0,0,0,0],
#       [1,0,0,1,0,0,0,0],
#       [1,1,1,0,0,0,0,0],
#       [0,1,0,0,0,0,1,0],
#       [0,0,1,0,1,0,0,0],
#        [1,0,0,1,0,0,1,0]]

# DATA = [[1,0,0,0,0,0,0,0],
#         [1,0,0,0,0,0,0,0]]

# PEAK DETECTION VARIABLES
LAG_MOVING_STD = 50
THRESHOLD_SIGNAL_MOVING_STD = 3.5
INFLUENCE_SIGNAL_MOVING_STD = 0


# training
TRAINING_FILENAME = './training/training_data_3_states'
#AI_MODEL = './ai_pipeline_2_mixed_songs'
AI_MODEL = './ai_models/model_3_layers_3_states_4'

HIGH_BITS_PLACES_FILE = './training/high_bit_places_pickle'
ENCODED_PLACES_FILE = './training/encoded_places_file'

HIGH_BIT_PLACES = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, \
                230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 260,\
                261, 262, 263, 264, 265, 266, 267, 268, 269, 290, 291,\
                292, 293, 294, 295, 296, 297, 298, 299, 320, 321, 322, \
                323, 324, 325, 326, 327, 328, 329]


"""
[195, 196, 197, 198, 199, 210, 211, 212, 213, 214, 225, 226, 227, 228, 229, 240, 241, 242, 243, 244, 255, 256, 257, 258, 259, 270, 271, 272, 273, 274, 285, 286, 287, 288, 289, 300, 301, 302, 303, 304, 315, 316, 317, 318, 319, 330, 331, 332, 333, 334]
"""


'''____________'''

TOTAL_BITS = 8
PHASE_JUMP_1 = np.pi*0.9
PHASE_JUMP_0 = 0
