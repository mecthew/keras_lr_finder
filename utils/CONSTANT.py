UNBALANCE_RATIO = 3

# symbols
UNEVEN = 'uneven'
MIN_CLASS = 'min_class'
MAX_CLASS = 'max_class'
CLASS_NUM = 'class_num'
EACH_CLASS_NUM = 'each_class_num'
NUM = 'num'
INDEX = 'index'
INDICES = 'indices'
CLASS_INFO = 'each_class_info'
SAMPLE_INFO = "sample_info"
ITER_NUM = 'iter_num'
TRAIN_NUM = 'train_num'
TEST_NUM = 'test_num'

# CONSTANT NUMBER
MAX_SAMPLE_NUM = 3000
MIN_SAMPLE_NUM = 1000
MAX_VALID_PERCLASS_SAMPLE = 100
MAX_VALID_SET_SIZE = 300
MIN_VALID_PER_CLASS = 1

# MODEL HYPER PARAMETERS
MODEL_FIRST_MAX_RUN_LOOP = 1
MODEL_MAX_RUN_LOOP = 20

# Modified by qmc
NUM_MFCC = 96  # num of mfcc features, default value is 24
MAX_AUDIO_DURATION = 5  # limited length of audio, like 20s
IS_CUT_AUDIO = True
AUDIO_SAMPLE_RATE = 16000
MAX_FRAME_NUM = 700
FFT_DURATION = 0.1
HOP_DURATION = 0.04

"""
metadata: a dict formed like:
    {"class_num": 10,
     "num_train_instances": 10000,
     "num_test_instances": 1000,
     "time_budget": 300,
     "indices": [[1, 2], [3, 6], [4, 5]...],
     "uneven": True / False,
     "each_class_info": [
         {"indices": [1, 2],
          "num": 2
          },
         ...
     ],
     "min_class": {
         "index": 0,
         "num": 2
     },
     "max_class": {
         "index": 5,
         "num": 10
     },
     "sample_info": "a bool `numpy.ndarray` matrix of shape(sample_count, 1)"
     }
"""
