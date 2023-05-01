import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
from DeepMapping import deepmapping
from DeepMapping.ndb_utils import df_preprocess, data_manipulation
from tqdm.auto import tqdm

# searched model structure, which is the output of NAS algorithm
SEARCH_MODEL_STRUCTURE = {'l1': 900, 'l1_input': 0, 'task0_input': 1}
# set to True if the model is searched for data manipulation Task
IS_DATA_MANIPULATION = True
# data set name
TASK_NAME = 'data_manipulation/single_column_high_corr_100m'
# number of epochs of the training model
NUM_EPOCHS = 2000

benchmark = 'deepmapping'
df = pd.read_csv('dataset/{}.csv'.format(TASK_NAME))
df, data_ori = df_preprocess(df, benchmark)
if IS_DATA_MANIPULATION:
    df, data_ori = data_manipulation(df, ops='Default')

model_path_dir = os.path.join('models', 'nas', TASK_NAME.split('/')[0])
os.makedirs(model_path_dir, exist_ok=True)    
model_path_to_save = os.path.join('models', 'nas', TASK_NAME+ '.h5')

# train the model
model, train_history = deepmapping.compress_data(df, SEARCH_MODEL_STRUCTURE, NUM_EPOCHS=NUM_EPOCHS)
model.save(model_path_to_save, include_optimizer=False)

print("[INFO] Trained model is saved to: {}".format(model_path_to_save))