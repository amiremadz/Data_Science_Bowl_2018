import os

def init():
    global train_ids
    global test_ids
    global TRAIN_PATH
    global TEST_PATH

    # Data Path
    TRAIN_PATH = '../data/stage1_train/'
    TEST_PATH  = '../data/stage1_test/'

    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
