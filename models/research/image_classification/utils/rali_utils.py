RALI_TRAIN_ENUM = None
RALI_VAL_ENUM = None

def initialize_enumerator(iterator, iterator_type):
    if (iterator_type == 0):
        global RALI_TRAIN_ENUM
        RALI_TRAIN_ENUM = enumerate(iterator, 0)
    elif (iterator_type == 1):
        global RALI_VAL_ENUM
        RALI_VAL_ENUM = enumerate(iterator, 0)