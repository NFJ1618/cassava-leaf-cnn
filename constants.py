EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = (224,224)
TEST_RATIO = 0.1
EVAL_PER_EPOCH = 3
DATA_RATIO = 0.1
NUM_IMAGES = 20000
N_EVAL = ((DATA_RATIO*NUM_IMAGES*(1-TEST_RATIO))/BATCH_SIZE)//EVAL_PER_EPOCH