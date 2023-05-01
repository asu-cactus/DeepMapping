from DeepMapping import nas_search_model

NUM_ITER = 2000 # number of total search iteration
NUM_EPOCHS_MODEL_TRAIN = 5 # number of model training epochs per model training iteration
NUM_ITR_CTRL_TRAIN = 50 # the gap between the iteration the controller will get trained
IS_DATA_MANIPULATION = True # set to True if the model is searched for data manipulation Task

TASK_NAME = 'nas' # task name
ALPHA = 1 
CUDA_ID = 0 # cuda id
DATA_SUB_PATH = 'data_manipulation/single_column_high_corr_100m' # path to dataset for NAS, you can also try others, like tpch-s1/customer, tpcds-s1/customer_demographics

searched_model_structure = nas_search_model.search_model(DATA_SUB_PATH=DATA_SUB_PATH, NUM_ITER=NUM_ITER,
                                                         NUM_EPOCHS_MODEL_TRAIN=NUM_EPOCHS_MODEL_TRAIN,
                                                         NUM_ITR_CTRL_TRAIN=NUM_ITR_CTRL_TRAIN,
                                                         TASK_NAME=TASK_NAME, CUDA_ID=CUDA_ID, IS_DATA_MANIPULATION=IS_DATA_MANIPULATION)

# the searched model will be printout
print(searched_model_structure)