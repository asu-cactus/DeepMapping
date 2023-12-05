import ctypes
import gc
import math
import numpy as np
import onnx
import onnxruntime as ort
import os
import pandas as pd 
import sys
import tensorflow as tf
import zstd
import multiprocessing
import concurrent.futures
import multiprocessing.shared_memory as shm

from bitarray import bitarray
from collections import defaultdict
from DeepMapping import ndb_utils
from onnx_opcounter import calculate_params

from sklearn import preprocessing
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from tqdm.auto import tqdm


ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.bool_, 
                                            ndim=1,
                                            flags="C")
ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, 
                                    ndim=1,
                                    flags="C")

shared_utils = ctypes.CDLL(os.path.abspath("shared_utils.so")) # Or full path to file 
shared_utils.create_fetures.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int]
shared_utils.create_fetures_mutlt_thread_mgr.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int32, ctypes.c_int32]

shared_utils.aux_look_up_bin.argtypes = [ND_POINTER_2, ctypes.c_int, ctypes.c_long]
shared_utils.aux_look_up_bin.restype = ctypes.c_long

def encode_label(arr):
    label_encoder = preprocessing.LabelEncoder().fit(arr)
    arr_encode = label_encoder.transform(arr)
    return arr_encode, label_encoder

def create_features_c_multi_thread(shared_utils, x, num_record, max_len, num_threads=4):
    # feature extraction, multi-thread implementation
    x_features_arr = np.zeros(num_record * max_len * 10, dtype=np.bool_)
    x_features_arr_ptr = shared_utils.create_fetures_mutlt_thread_mgr(x_features_arr, x, num_record, max_len, num_threads)
    features = np.frombuffer(x_features_arr_ptr.contents, dtype=np.bool_).reshape(num_record, -1).copy()
    return features
        
def create_features_c_single_thread(shared_utils, x, num_record, max_len):
    # feature extraction, single-thread implementation
    x_features_arr = np.zeros(num_record * max_len * 10, dtype=np.bool_)
    x_features_arr_ptr = shared_utils.create_fetures(x_features_arr, x, num_record, max_len)
    x_features_sampled = np.frombuffer(x_features_arr_ptr.contents, dtype=np.bool_).reshape(num_record, -1).copy()
    return x_features_sampled

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, list_y, batch_size, max_len, shuffle=True, num_threads=4):
        self.x = x
        self.list_y = list_y
        self.num_task = len(list_y)
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.num_threads = num_threads
    
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, index):
        Y = dict()
        idx_start = index*self.batch_size
        idx_end = (index+1)*self.batch_size
        data_x = self.x[idx_start:idx_end]
        for i in range(self.num_task):
            task_name = 'task{}'.format(i)
            Y[task_name] = self.list_y[i][idx_start:idx_end]

        num_record = len(data_x)
        max_len = self.max_len
        
        shared_utils.create_fetures.restype = ctypes.POINTER(ctypes.c_bool * (num_record * max_len * 10))
        shared_utils.create_fetures_mutlt_thread_mgr.restype = ctypes.POINTER(ctypes.c_bool * (num_record * max_len * 10))

        X = create_features_c_multi_thread(shared_utils, data_x, num_record, max_len, self.num_threads)
        
        return X, Y
    
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class InferenceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, batch_size, max_len, shuffle=False, num_threads=4):
        self.x = x
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.num_threads = num_threads
    
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, index):
        idx_start = index*self.batch_size
        idx_end = (index+1)*self.batch_size
        data_x = self.x[idx_start:idx_end]

        num_record = len(data_x)
        max_len = self.max_len
        
        shared_utils.create_fetures.restype = ctypes.POINTER(ctypes.c_bool * (num_record * max_len * 10))
        shared_utils.create_fetures_mutlt_thread_mgr.restype = ctypes.POINTER(ctypes.c_bool * (num_record * max_len * 10))

        X = create_features_c_multi_thread(shared_utils, data_x, num_record, max_len, self.num_threads)
        
        return X
    
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.x))


def build_model(num_in, model_sturcture, list_num_out):
    x = tf.keras.Input(shape=(num_in,1))
    flatten = tf.keras.layers.Flatten(input_shape=(num_in,1), name='in')(x)
    shared = flatten
    list_output_layers = []
    model_layers = dict()
    layers_output = dict()
    layers_output[0] = shared
    list_outs = []

    for structure_name, structure_value in model_sturcture.items():

        if 'input' not in structure_name and 'task' not in structure_name:
            # define a layer
            model_layers[structure_name] = tf.keras.layers.Dense(units=structure_value, activation='relu', kernel_regularizer=regularizers.L2(l2=0), name=structure_name)
        elif 'input' in structure_name and 'task' not in structure_name:
            # define a layer's input
            layer_name = structure_name.split('_')[0]
            layer_index = int(str(layer_name[1:]))
            layers_output[layer_index] = model_layers[layer_name](layers_output[structure_value])
        elif 'input' in structure_name and 'task' in structure_name:
            # define task output layer and compute output
            layer_name = structure_name.split('_')[0]
            task_idx = int(str(layer_name[4:]))
            model_layers[layer_name] = tf.keras.layers.Dense(units=list_num_out[task_idx], activation='softmax', kernel_regularizer=regularizers.L2(l2=0), name=layer_name)
            task_output = model_layers[layer_name](layers_output[structure_value])
            list_outs.append(task_output)
    model = tf.keras.Model(inputs=x, outputs=list_outs,
                                name='muilti-class') 
    return model


class SOMT(keras.callbacks.Callback):
    def __init__(self, model,  train_thold):
        super(SOMT, self).__init__()
        self.model=model        
        self.train_thold=train_thold
        # self.valid_thold=valid_thold
        
    def on_epoch_end(self,epoch, logs=None): 
        ep=epoch+1
        msg = ""
        train_flag = False 
        for k,v in logs.items():
          msg += str(k) + ": {:.2f} ".format(v)
          if 'accuracy' in k and train_flag == False:
            if v < self.train_thold:
              train_flag = True
        if train_flag == False:
          self.model.stop_training = True

def compress_data(df, model_sturcture, batch_size=1024, num_epochs=500, train_verbose=1, train=True):
    df_key = [df.columns[0]]
    list_y_encoded = []
    list_y_encoder = []

    for col in df.columns:
        if col not in df_key:
            encoded_val, encoder = encode_label(df[col])
            list_y_encoded.append(encoded_val)
            list_y_encoder.append(encoder)
    num_tasks = len(list_y_encoded)
    num_tasks

    for encoder in list_y_encoder:
        print(len(encoder.classes_))
    
    x = df[df_key[0]].values.astype(np.int32)
    max_len = len(str(np.max(x)))
    list_num_out = [len(encoder.classes_) for encoder in list_y_encoder]
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = build_model(max_len*10, model_sturcture, list_num_out)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3/1000)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    train_generator = DataGenerator(x, list_y_encoded, batch_size, max_len)

    if train == True:
        train_history = model.fit(train_generator, epochs=num_epochs, verbose=train_verbose, callbacks=[SOMT(model, 1)])
        return model, train_history
    else:
        return model, train_generator

def finetune_model(df, model_path, batch_size=1024, num_epochs=500, train_verbose=1, train=True):
    df_key = [df.columns[0]]
    list_y_encoded = []
    list_y_encoder = []

    for col in df.columns:
        if col not in df_key:
            encoded_val, encoder = encode_label(df[col])
            list_y_encoded.append(encoded_val)
            list_y_encoder.append(encoder)
    num_tasks = len(list_y_encoded)
    num_tasks

    for encoder in list_y_encoder:
        print(len(encoder.classes_))
    
    x = df[df_key[0]].values.astype(np.int32)
    max_len = len(str(np.max(x)))
    print('MAX LEN', max_len)
    list_num_out = [len(encoder.classes_) for encoder in list_y_encoder]
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    early_stopping = EarlyStopping(monitor='loss',  # Metric to monitor for early stopping
                               min_delta=0.0005,
                               patience=3,           # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored metric

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        # model = build_model(max_len*10, model_sturcture, list_num_out)
        # x = tf.keras.Input(shape=(num_in,1))
        model = tf.keras.models.load_model(model_path)

        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3/1000)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    train_generator = DataGenerator(x, list_y_encoded, batch_size, max_len)
    
    if train == True:
        train_timer = ndb_utils.Timer()
        train_timer.tic()
        train_history = model.fit(train_generator, epochs=num_epochs, verbose=train_verbose, callbacks=[SOMT(model, 1), early_stopping])
        t_nn_train = train_timer.toc()/1000
        print('[INFO] NN Fine-tune Time: {} sec'.format(t_nn_train))
        return model, train_history
    else:
        return model, train_generator
    
def write_partition_to_disk(arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, num_record, num_cols, block_start_idx, block_end_idx, comp_data_dir, zstd_compress_level):
    # Open the shared memory array
    arr_shm = shm.SharedMemory(name=arr_shm_name)
    # Access the shared array as a NumPy array
    data_ori = np.ndarray((num_record, num_cols), dtype=arr_shm_dtypes, buffer=arr_shm.buf)

    arr_shm1 = shm.SharedMemory(name=arr_shm_name1)
    data_partition_idx = np.ndarray((num_record, ), dtype=arr_shm_dtypes1, buffer=arr_shm1.buf)

    total_size = 0
    enable_progress_bar = block_end_idx >= np.max(data_partition_idx)
    if enable_progress_bar:
        progress_bar = tqdm(range(block_start_idx, block_end_idx), desc="Progress")

    for block_idx in range(block_start_idx, block_end_idx):
        data_idx = data_partition_idx == block_idx
        data_part = data_ori[data_idx]
        if len(data_part) == 0:
            continue
        data_bytes = data_part.tobytes()
        data_zstd_comp = zstd.compress(data_bytes, zstd_compress_level)
        file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')
        ndb_utils.save_byte_to_disk(file_name, data_zstd_comp)
        total_size += sys.getsizeof(data_zstd_comp)/1024/1024
        if enable_progress_bar:
            progress_bar.update(1)
    return total_size

def update_partition_to_disk(arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, arr_shm_name2, arr_shm_dtypes2, arr_shm_name3, arr_shm_dtypes3, num_record, num_cols, block_start_idx, block_end_idx, comp_data_dir, zstd_compress_level, data_ops):
    # Open the shared memory array
    arr_shm = shm.SharedMemory(name=arr_shm_name)
    # Access the shared array as a NumPy array
    data_ori = np.ndarray((num_record, num_cols), dtype=arr_shm_dtypes, buffer=arr_shm.buf)

    arr_shm1 = shm.SharedMemory(name=arr_shm_name1)
    data_partition_idx = np.ndarray((num_record, ), dtype=arr_shm_dtypes1, buffer=arr_shm1.buf)

    arr_shm2 = shm.SharedMemory(name=arr_shm_name2)
    correct_pred_index = np.ndarray((num_record, ), dtype=arr_shm_dtypes2, buffer=arr_shm2.buf)

    arr_shm3 = shm.SharedMemory(name=arr_shm_name3)
    correct_pred_partition_idx = np.ndarray((num_record, ), dtype=arr_shm_dtypes3, buffer=arr_shm3.buf)

    timer = ndb_utils.Timer()

    total_size = 0
    enable_progress_bar = block_end_idx >= np.max(data_partition_idx)
    if enable_progress_bar:
        progress_bar = tqdm(range(block_start_idx, block_end_idx), desc="Progress")

    for block_idx in range(block_start_idx, block_end_idx):
        data_idx = data_partition_idx == block_idx
        data_part = data_ori[data_idx]
        if len(data_part) == 0:
            continue
        file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')
        try:
            block_zstd_comp = ndb_utils.read_bytes_from_disk(file_name)
        except: 
            sorted_indices = np.argsort(data_part[:, 0])
            data_part = data_part[sorted_indices]
            data_bytes = data_part.tobytes()
            data_zstd_comp = zstd.compress(data_bytes, zstd_compress_level)
            with ndb_utils.get_temp_file_path() as temp_path:
                ndb_utils.save_byte_to_disk(temp_path, data_zstd_comp)
            continue

        data_uncomp = np.frombuffer(zstd.decompress(block_zstd_comp), dtype=np.int32).reshape(-1, num_cols)
        # insert or update the misclassified ones to the partition
        for i in range(len(data_part)):
            key = data_part[i,0]
            searched_index = ndb_utils.binary_search(data_uncomp[:,0], key, len(data_uncomp))
            if searched_index != -1:
                data_uncomp[searched_index] = tuple(data_part[i])
            else:
                data_uncomp = np.concatenate([data_uncomp, data_part[i].reshape(1,-1)], axis=0)
        
        if data_ops == 'Update':
            data_idx = correct_pred_partition_idx == block_idx
            correct_partition_pred_index = correct_pred_index[data_idx]
            for i in range(len(correct_partition_pred_index)):
                searched_index = ndb_utils.binary_search(data_uncomp[:,0], correct_partition_pred_index[i], len(data_uncomp))
                if searched_index != -1:
                    data_uncomp = np.delete(data_uncomp, searched_index, axis=0)

        data_bytes = data_part.tobytes()
        data_zstd_comp = zstd.compress(data_bytes, zstd_compress_level)
        with ndb_utils.get_temp_file_path() as temp_path:
            ndb_utils.save_byte_to_disk(temp_path, data_zstd_comp)
        if enable_progress_bar:
            progress_bar.update(1)
    return total_size

def delete_tuple_in_disk(arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, num_record, num_cols, block_start_idx, block_end_idx, comp_data_dir, zstd_compress_level, data_ops):
    # Open the shared memory array
    arr_shm = shm.SharedMemory(name=arr_shm_name)
    # Access the shared array as a NumPy array
    data_idx_to_delete = np.ndarray((num_record, num_cols), dtype=arr_shm_dtypes, buffer=arr_shm.buf)

    arr_shm1 = shm.SharedMemory(name=arr_shm_name1)
    data_idx_to_delete_partition_idx = np.ndarray((num_record, ), dtype=arr_shm_dtypes1, buffer=arr_shm1.buf)

    total_size = 0
    enable_progress_bar = block_end_idx >= np.max(data_idx_to_delete_partition_idx)
    if enable_progress_bar:
        progress_bar = tqdm(range(block_start_idx, block_end_idx), desc="Progress")

    for block_idx in range(block_start_idx, block_end_idx):
        data_idx = data_idx_to_delete_partition_idx == block_idx

        data_part = data_idx_to_delete[data_idx]
        if len(data_part) == 0:
            continue
        file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')
        try:
            block_zstd_comp = ndb_utils.read_bytes_from_disk(file_name)
        except:
            continue
        data_uncomp = np.frombuffer(zstd.decompress(block_zstd_comp), dtype=np.int32).reshape(-1, num_cols)
        # insert or update the misclassified ones to the partition
        for i in range(len(data_part)):
            key = data_part[i,0]
            searched_index = ndb_utils.binary_search(data_uncomp[:,0], key, len(data_uncomp))
            if searched_index != -1:
                data_uncomp = np.delete(data_uncomp, searched_index, axis=0)

        data_bytes = data_part.tobytes()
        data_zstd_comp = zstd.compress(data_bytes, zstd_compress_level)
        with ndb_utils.get_temp_file_path() as temp_path:
            ndb_utils.save_byte_to_disk(temp_path, data_zstd_comp)
        if enable_progress_bar:
            progress_bar.update(1)
    return total_size


def measure_latency_any(df, data_ori, task_name, sample_size, 
                    generate_file=True,
                    num_loop=10, num_query=1, search_algo='binary', path_to_model=None, root_path='temp', model_root='models',
                    block_size=1024*1024, **kwargs):
    """Measure the end-end latency of data query

    Args:
        df : dataframe
            dataset in pd.dataframe format
        data_ori : np.record
            dataset in np.record format
        task_name : str
            task name
        sample_size : int
            number of queried data per query
        generate_file : bool
            whether need to store the data to disk
        num_loop : int
            number of loops to run for measuring the latency
        num_query : int
            number of queries
        search_algo : str
            search algorithm that applied to search entry in each partition
        path_to_model : str
            load model from custom path
        block_size  : int
            block size for each partition, size in bytes
    """
    if 'zstd_compress_level' in kwargs:
        zstd_compress_level = kwargs['zstd_compress_level']
    max_generate_file_threads = int(os.environ['MAX_GENERATE_FILE_THREADS'])
    backend = os.environ['BACKEND']
    mode = os.environ['MODE']
    data_ori_size = 0
    data_comp_size = 0
    arr_latency = None 
    arr_result = None

    
    # root_path = 'temp'
    if os.environ['MODE'] == 'tuning':
        root_path = 'temp_tune'
    folder_name = 'deepmapping'
    comp_data_dir = os.path.join(root_path, task_name, folder_name, str(block_size))
    if 'DATA_OPS' in os.environ and 'CHANGE_RATIO' in os.environ:
        comp_data_dir = os.path.join(comp_data_dir, os.environ['DATA_OPS'], os.environ['CHANGE_RATIO'])

    print('[Generate File Path]: {}'.format(comp_data_dir))
    
    df_key = [df.columns[0]]
    list_y_encoded = []
    list_y_encoder = []
    dict_contigous_key= []
    size_encoder = 0
    exist_bit_arr = None
    num_record_per_part = None 
    data_comp_size = None 
    max_len = None 
    x_start = None 
    x_end = None 
    num_tasks = None 

    shared_utils = ctypes.CDLL(os.path.abspath("shared_utils.so")) # Or full path to file 
    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.bool_, 
                                            ndim=1,
                                            flags="C")
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, 
                                        ndim=1,
                                        flags="C")
    shared_utils.create_fetures.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int]
    shared_utils.create_fetures_mutlt_thread_mgr.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int32, ctypes.c_int32]
    shared_utils.aux_look_up_bin.argtypes = [ND_POINTER_2, ctypes.c_int, ctypes.c_long]
    shared_utils.aux_look_up_bin.restype = ctypes.c_long

    num_threads = 4
    if sample_size <= 4:
        num_threads = 1
    
    if path_to_model is None:
        if backend == 'tf':
            model = tf.keras.models.load_model('{}/nas/{}.h5'.format(model_root, task_name), compile=False)
            model_size = model.count_params()*4/1024/1024
        elif backend == 'onnx': 
            model = ort.InferenceSession('{}/nas/{}.onnx'.format(model_root, task_name), providers=['CUDAExecutionProvider']) 
            input_name = model.get_inputs()[0].name
            model_size = calculate_params(onnx.load_model('{}/nas/{}.onnx'.format(model_root, task_name)))*4/1024/1024
    else:
        if backend == 'tf':
            model = tf.keras.models.load_model(path_to_model, compile=False)
            model_size = model.count_params()*4/1024/1024
        elif backend == 'onnx':
            model = ort.InferenceSession(path_to_model, providers=['CUDAExecutionProvider']) 
            input_name = model.get_inputs()[0].name
            model_size = calculate_params(onnx.load_model(path_to_model))*4/1024/1024
        print('[INFO] loaded model from custom path: {}'.format(path_to_model))
    
    # max_len = len(str(np.max(x)))
    if backend == 'tf':
        max_len = int(model.layers[0].input_shape[0][1]/10)
    elif backend == 'onnx':
        max_len = int(model.get_inputs()[0].shape[1]/10)
    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        exp_data_dict = dict()


        for col in df.columns:
            if col not in df_key:
                encoded_val, encoder = encode_label(df[col])
                list_y_encoded.append(encoded_val)
                list_y_encoder.append(encoder)
                size_encoder += encoder.classes_.nbytes / 1024 / 1024
        num_tasks = len(list_y_encoded)

        exp_data_dict['list_y_encoder'] = list_y_encoder 
        exp_data_dict['num_tasks'] = num_tasks 
        exp_data_dict['size_encoder'] = size_encoder 
        
        for encoder in list_y_encoder:
            print(len(encoder.classes_))

        x = df[df_key[0]].values.astype(np.int32)
        exp_data_dict['max_len'] = max_len
        y = np.array(list_y_encoded).T.astype(np.int32)
        data = np.concatenate((x.reshape(-1,1), y), axis=1, dtype=np.int32)
        print(data.nbytes/1024/1024)

        train_generator = DataGenerator(x, list_y_encoded, 1024*2**4, max_len, False, num_threads)

        # exist_bitarray
        x_start = np.min(x)
        x_end = np.max(x)
        exist_bit_arr = bitarray('0')*(x_end - x_start + 1)

        for val in x:
            exist_bit_arr[val-x_start] = 1
        print(sys.getsizeof(exist_bit_arr)/1024/1024)

        misclassified_index = []

        for idx, (x_sub,y_sub) in tqdm(enumerate(train_generator), total=len(train_generator)):
            y_sub = list(y_sub.values())
            if backend == 'tf':
                y_sub_pred = model(x_sub)
            elif backend == 'onnx':
                 y_sub_pred = model.run(None, {input_name: np.expand_dims(x_sub, -1).astype(np.float32)})
            mis_pred = []

            for i in range(num_tasks):
                if num_tasks == 1:
                    if backend == 'tf':
                        mis_pred.append(y_sub[i] != np.argmax(y_sub_pred, axis=1))
                    if backend == 'onnx':
                        mis_pred.append(y_sub[i] != np.argmax(y_sub_pred[0], axis=1))
                    # mis_pred.append(y_sub[i] != np.argmax(y_sub_pred, axis=1))
                else:
                    mis_pred.append(y_sub[i] != np.argmax(y_sub_pred[i], axis=1))
            
            misclassified = None

            for val in mis_pred:
                if misclassified is None:
                        misclassified = val
                else:
                    misclassified = np.logical_or(misclassified, val)
                    
            misclassified_index.extend(misclassified)

        misclassified_data = data[misclassified_index]
        misclassified_data_comp = zstd.compress(misclassified_data.tobytes())
        print(sys.getsizeof(misclassified_data_comp)/1024/1024)

        # partition
        if len(misclassified_data) == 0:
            misclassified_data = np.zeros((1,2))
        record_size = misclassified_data[0].nbytes
        num_record_per_part = np.floor(block_size / record_size)

        x_start = np.min(misclassified_data[:,0])
        x_end = np.max(misclassified_data[:,0])
        x_range = x_end - x_start
        num_partition = int(math.ceil(x_range / num_record_per_part))
        print(record_size*num_record_per_part/1024, num_partition)

        list_comp_aux_blocks = []
        comp_zstd_size = 0
        data_partition_idx = (misclassified_data[:, 0] - x_start) // num_record_per_part

        num_threads_to_generate_file = max_generate_file_threads if max_generate_file_threads <= ndb_utils.get_sys_num_threads() else ndb_utils.get_sys_num_threads()
        num_partition_per_core = int(np.ceil(num_partition/num_threads_to_generate_file))
        arr_shm = shm.SharedMemory(create=True, size=misclassified_data.nbytes)
        shared_array = np.ndarray(misclassified_data.shape, dtype=misclassified_data.dtype, buffer=arr_shm.buf)
        shared_array[:] = misclassified_data[:]
        arr_shm_name = arr_shm.name
        arr_shm_dtypes = misclassified_data.dtype

        arr_shm1 = shm.SharedMemory(create=True, size=data_partition_idx.nbytes)
        shared_array1 = np.ndarray(data_partition_idx.shape, dtype=data_partition_idx.dtype, buffer=arr_shm1.buf)
        shared_array1[:] = data_partition_idx[:]
        arr_shm_name1 = arr_shm1.name
        arr_shm_dtypes1 = data_partition_idx.dtype

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads_to_generate_file) as executor:
            # Launch threads to process the numbers
            futures = [executor.submit(write_partition_to_disk, arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, misclassified_data.shape[0], misclassified_data.shape[1], i*num_partition_per_core, (i+1)*num_partition_per_core, comp_data_dir, zstd_compress_level) for i in range(num_threads_to_generate_file)]
            # Retrieve the results from each thread
            for future in concurrent.futures.as_completed(futures):
                comp_zstd_size += future.result()

        arr_shm.close()
        arr_shm.unlink()
        arr_shm1.close()
        arr_shm1.unlink()

        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = [size_encoder, comp_zstd_size, model_size, sys.getsizeof(zstd.compress(exist_bit_arr.tobytes()))/1024/1024]
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_comp_size))
        x = df[df_key[0]].values.astype(np.int32)
        x_start = np.min(x)
        x_end = np.max(x)

        exp_data_dict['num_record_per_part'] = num_record_per_part 
        exp_data_dict['data_ori_size'] = data_ori_size
        exp_data_dict['data_comp_size'] = [size_encoder, comp_zstd_size, model_size, sys.getsizeof(zstd.compress(exist_bit_arr.tobytes()))/1024/1024]
        exp_data_dict['max_len'] = max_len 
        exp_data_dict['x_start'] = x_start 
        exp_data_dict['x_end'] = x_end 
        ndb_utils.save_byte_to_disk(os.path.join(comp_data_dir, 'exist_bit_arr.data'), zstd.compress(exist_bit_arr.tobytes()))
        ndb_utils.save_obj_to_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'), exp_data_dict)
        list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
        
    else:
        exist_bit_arr = bitarray()
        exist_bit_arr.frombytes(zstd.decompress(ndb_utils.read_bytes_from_disk(os.path.join(comp_data_dir, 'exist_bit_arr.data'))))

        exp_data_dict = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'))
        num_record_per_part = exp_data_dict['num_record_per_part']
        data_ori_size = exp_data_dict['data_ori_size']
        data_comp_size = exp_data_dict['data_comp_size']
        max_len = exp_data_dict['max_len']
        x_start = exp_data_dict['x_start']
        x_end = exp_data_dict['x_end']
        list_y_encoder = exp_data_dict['list_y_encoder']
        num_tasks = exp_data_dict['num_tasks']
        size_encoder = exp_data_dict['size_encoder']
        list_sample_index = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(root_path, task_name, 'sample_index_{}.data'.format(sample_size)))
    
    data_ori = data_ori[:2]
    del df 
    gc.collect() 
    shared_utils.create_fetures.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int]
    shared_utils.create_fetures_mutlt_thread_mgr.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int32, ctypes.c_int32]
    shared_utils.create_fetures_mutlt_thread_mgr.restype = ctypes.POINTER(ctypes.c_bool * (sample_size * max_len * 10))

    timer_creatfeatures = ndb_utils.Timer()
    timer_locate_part = ndb_utils.Timer()
    timer_nn = ndb_utils.Timer()
    timer_aux_lookup = ndb_utils.Timer()
    timer_total = ndb_utils.Timer()
    timer_decomp = ndb_utils.Timer()
    timer_exist_lookup = ndb_utils.Timer()
    timer_remap = ndb_utils.Timer()
    timer_sort = ndb_utils.Timer()
    timer_build_index = ndb_utils.Timer()
    t_remap = 0
    t_decomp = 0
    t_createfeatures = 0
    t_aux_lookup = 0
    t_nn = 0
    t_exist_lookup = 0
    t_total = 0
    t_sort = 0
    t_locate_part = 0
    t_build_index = 0
    block_bytes_size = 0
    timer_total.tic()
    for _ in tqdm(range(num_loop)):
        partition_hit = dict()
        decomp_aux_block = dict()
        num_decomp = 0
        count_nonexist = 0
        peak_memory = 0
        cache_block_memory = 0
        gc.collect()

        # build hash table
        if search_algo == 'hash':
            data_hash = dict()
        for query_idx in range(num_query):
            sample_index = list_sample_index[query_idx]                
            timer_total.tic()                
            timer_sort.tic()
            sample_index_sorted = np.sort(sample_index)
            sample_index_argsort = np.argsort(sample_index)
            sample_index_partition = (sample_index_sorted - x_start) // num_record_per_part
            sample_index_partition = sample_index_partition.astype(np.int32)
            t_sort += timer_sort.toc()     
            result = np.ndarray((sample_size, ), dtype=data_ori.dtype)
            result[df_key[0]] = sample_index
            if mode == 'edge':
                edge_batch_size = 5000
                timer_creatfeatures.tic()       
                inference_generator = InferenceDataGenerator(sample_index, edge_batch_size, max_len, False, num_threads)
                for idx, (x_sub) in enumerate(inference_generator):
                    t_createfeatures += timer_creatfeatures.toc()
                    timer_nn.tic()
                    if backend == 'tf':
                        y_nn_pred = model(x_sub)
                    elif backend == 'onnx':
                        y_nn_pred = model.run(None, {input_name: np.expand_dims(x_sub, -1).astype(np.float32)})
                    
                    for i in range(num_tasks):
                        if num_tasks == 1 and backend == 'onnx':
                            col_name = data_ori.dtype.names[i+1]
                            result[col_name][idx*edge_batch_size:(idx+1)*edge_batch_size] = np.argmax(y_nn_pred[0], axis=1)
                        elif num_tasks == 1 and backend == 'tf':
                            col_name = data_ori.dtype.names[i+1]
                            result[col_name][idx*edge_batch_size:(idx+1)*edge_batch_size] = np.argmax(y_nn_pred, axis=1)
                        else:
                            col_name = data_ori.dtype.names[i+1]
                            result[col_name][idx*edge_batch_size:(idx+1)*edge_batch_size] = np.argmax(y_nn_pred[i], axis=1)
                    t_nn += timer_nn.toc()
                    timer_creatfeatures.tic()  

            else:                         
                timer_creatfeatures.tic()        

                x_features_arr = np.zeros(sample_size * max_len * 10, dtype=bool)
                x_features_arr_ptr = shared_utils.create_fetures_mutlt_thread_mgr(
                    x_features_arr, sample_index, sample_size, max_len, num_threads)
                sampled_features = np.frombuffer(
                    x_features_arr_ptr.contents, dtype=bool).reshape(sample_size, -1)
                # sampled_features = ndb_utils.create_features(sample_index, max_len)[0]
                t_createfeatures += timer_creatfeatures.toc()
                timer_nn.tic()
                if backend == 'tf':
                    y_nn_pred = model(sampled_features)
                elif backend == 'onnx':
                    y_nn_pred = model.run(None, {input_name: np.expand_dims(sampled_features, -1).astype(np.float32)})
                
                for i in range(num_tasks):
                    if num_tasks == 1 and backend == 'onnx':
                        col_name = data_ori.dtype.names[i+1]
                        result[col_name] = np.argmax(y_nn_pred[0], axis=1)
                    elif num_tasks == 1 and backend == 'tf':
                        col_name = data_ori.dtype.names[i+1]
                        result[col_name] = np.argmax(y_nn_pred, axis=1)
                    else:
                        col_name = data_ori.dtype.names[i+1]
                        result[col_name] = np.argmax(y_nn_pred[i], axis=1)
                t_nn += timer_nn.toc()
            
            for idx, val in enumerate(sample_index):
                # ------ non exist look up
                timer_exist_lookup.tic()
                query_key = sample_index_sorted[idx]
                query_key_index_in_old = sample_index_argsort[idx]
                exist_flag = exist_bit_arr[query_key-x_start] == 1

                if not exist_flag:
                    result[query_key_index_in_old] = -1
                    count_nonexist += 1
                    t_exist_lookup += timer_exist_lookup.toc()
                else:
                    # misclassified lookup
                    t_exist_lookup += timer_exist_lookup.toc()
                    timer_locate_part.tic()
                    part_idx = sample_index_partition[idx]
                    t_locate_part += timer_locate_part.toc()
                    timer_decomp.tic()

                    if part_idx not in decomp_aux_block:
                        if mode == 'edge':
                            available_memory = ndb_utils.get_available_memory()
                            if available_memory < 1024*1024*100:
                                # memory not eneough, free some memory
                                decomp_aux_block = ndb_utils.evict_unused_partition(decomp_aux_block, partition_hit, free_memory=1024*1024*100)

                        partition_hit[part_idx] = 1


                        file_name = os.path.join(comp_data_dir, str(part_idx) + '.data')                           

                        if not os.path.exists(file_name):
                            continue
                        block_zstd_comp = ndb_utils.read_bytes_from_disk(file_name)
                        data_uncomp = np.frombuffer(
                        # zstd.decompress(block_zstd_comp), dtype=np.int32).reshape(-1, num_tasks+1).copy(order='F')
                        zstd.decompress(block_zstd_comp), dtype=np.int32).reshape(-1, num_tasks+1)
                        # decomp_aux_block[part_idx] = data_uncomp
                        try:
                            decomp_aux_block[part_idx] = data_uncomp
                        except:
                            decomp_aux_block = dict()
                            decomp_aux_block[part_idx] = data_uncomp
                        num_decomp += 1
                        block_bytes_size = sys.getsizeof(block_zstd_comp)
                        prev_part_idx = part_idx

                        if search_algo == 'hash':
                            t_decomp += timer_decomp.toc()
                            timer_build_index.tic()
                            for block_data_idx in range(len(data_uncomp)):
                                data_entry_key = data_uncomp[block_data_idx, 0]
                                # print(data_entry_key)
                                data_entry_val = data_uncomp[block_data_idx]
                                data_hash[data_entry_key] = data_entry_val   
                            cache_block_memory = sys.getsizeof(data_hash)
                            t_build_index += timer_build_index.toc()
                            timer_decomp.tic()
                        else:
                            cache_block_memory += data_uncomp.nbytes 
                    else:
                        data_uncomp = decomp_aux_block[part_idx]
                        partition_hit[part_idx] +=1
                    t_decomp += timer_decomp.toc()    
                    timer_aux_lookup.tic()
                    if search_algo == 'binary':
                        data_idx = ndb_utils.binary_search(data_uncomp[:,0], query_key, len(data_uncomp))
                        if data_idx != -1:
                            result[query_key_index_in_old] = tuple(data_uncomp[data_idx])
                        else:
                            count_nonexist += 1
                    elif search_algo == 'binary_c': 
                        data_idx = shared_utils.aux_look_up_bin(data_uncomp[:,0], query_key, len(data_uncomp))
                        if data_idx != -1:
                            result[query_key_index_in_old] = tuple(data_uncomp[data_idx])
                        else:
                            count_nonexist += 1
                    elif search_algo == 'hash':
                        if query_key in data_hash.keys():
                            result[query_key_index_in_old] = tuple(data_hash[query_key])

                    t_aux_lookup += timer_aux_lookup.toc()    

                if cache_block_memory + block_bytes_size > peak_memory:
                    peak_memory = cache_block_memory + block_bytes_size

            timer_remap.tic()
            for i in range(num_tasks):    
                col_name = data_ori.dtype.names[i+1]
                fun_a = lambda x: list_y_encoder[i].classes_[x]
                result[col_name] = fun_a(result[col_name].astype(np.int32))
            t_remap += timer_remap.toc()
            t_total += timer_total.toc()
        arr_result = result.copy()
        del result
        gc.collect()
    peak_memory += exist_bit_arr.nbytes
    arr_latency = np.array((data_ori_size, np.sum(data_comp_size), sample_size, 
            peak_memory/1024/1024, t_sort / num_loop, t_createfeatures / num_loop, t_nn / num_loop, t_locate_part / num_loop, t_decomp / num_loop, t_build_index / num_loop,
            t_aux_lookup / num_loop, t_exist_lookup / num_loop, t_remap / num_loop, t_total / num_loop, num_decomp, count_nonexist, exist_bit_arr.nbytes/1024/1024, model_size)).T

    return_latency = arr_latency.reshape((1,-1))

    return data_ori_size, data_comp_size, [arr_result], return_latency




def measure_latency_data_update(df, data_ori, data_change, data_op, task_name, sample_size, 
                    generate_file=True,
                    num_loop=10, num_query=5, search_algo='binary', path_to_model=None, root_path='temp', model_root='models',
                    block_size=1024*1024, **kwargs):
    """Measure the end-end latency of data query

    Args:
        df : dataframe
            dataset in pd.dataframe format
        data_ori : np.record
            dataset in np.record format
        task_name : str
            task name
        sample_size : int
            number of queried data per query
        generate_file : bool
            whether need to store the data to disk
        num_loop : int
            number of loops to run for measuring the latency
        num_query : int
            number of queries
        search_algo : str
            search algorithm that applied to search entry in each partition
        path_to_model : str
            load model from custom path
        block_size  : int
            block size for each partition, size in bytes
    """
    if 'zstd_compress_level' in kwargs:
        zstd_compress_level = kwargs['zstd_compress_level']
    max_generate_file_threads = int(os.environ['MAX_GENERATE_FILE_THREADS'])
    backend = os.environ['BACKEND']
    mode = os.environ['MODE']

    
    # root_path = 'temp'
    if os.environ['MODE'] == 'tuning':
        root_path = 'temp_tune'
    folder_name = 'deepmapping'
    comp_data_dir = os.path.join(root_path, task_name, folder_name, str(block_size))

    print('[Generate File Path]: {}'.format(comp_data_dir))
    
    list_y_encoder = []
    exist_bit_arr = None
    num_record_per_part = None 
    max_len = None 
    x_start = None 
    x_end = None 
    num_tasks = None 

    shared_utils = ctypes.CDLL(os.path.abspath("shared_utils.so")) # Or full path to file 
    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.bool_, 
                                            ndim=1,
                                            flags="C")
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, 
                                        ndim=1,
                                        flags="C")
    shared_utils.create_fetures.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int]
    shared_utils.create_fetures_mutlt_thread_mgr.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int32, ctypes.c_int32]
    shared_utils.aux_look_up_bin.argtypes = [ND_POINTER_2, ctypes.c_int, ctypes.c_long]
    shared_utils.aux_look_up_bin.restype = ctypes.c_long

    num_threads = 4
    if sample_size <= 4:
        num_threads = 1
    
    if path_to_model is None:
        if backend == 'tf':
            model = tf.keras.models.load_model('{}/nas/{}.h5'.format(model_root, task_name), compile=False)
            model_size = model.count_params()*4/1024/1024
        elif backend == 'onnx': 
            model = ort.InferenceSession('{}/nas/{}.onnx'.format(model_root, task_name), providers=['CUDAExecutionProvider']) 
            input_name = model.get_inputs()[0].name
            model_size = calculate_params(onnx.load_model('{}/nas/{}.onnx'.format(model_root, task_name)))*4/1024/1024
    else:
        if backend == 'tf':
            model = tf.keras.models.load_model(path_to_model, compile=False)
            model_size = model.count_params()*4/1024/1024
        elif backend == 'onnx':
            model = ort.InferenceSession(path_to_model, providers=['CUDAExecutionProvider']) 
            input_name = model.get_inputs()[0].name
            model_size = calculate_params(onnx.load_model(path_to_model))*4/1024/1024
        print('[INFO] loaded model from custom path: {}'.format(path_to_model))
    
    if backend == 'tf':
        max_len = int(model.layers[0].input_shape[0][1]/10)
    elif backend == 'onnx':
        max_len = int(model.get_inputs()[0].shape[1]/10)
    

    timer_update = ndb_utils.Timer()
    timer_encoding = ndb_utils.Timer()
    timer_nn = ndb_utils.Timer()
    timer_modify_aux = ndb_utils.Timer()

    t_encoding = 0
    t_nn = 0
    t_modify_uax = 0
    t_update = 0


    ori_data_change = data_change.copy()
    for _ in tqdm(range(num_loop)):
        data_change = ori_data_change.copy()
        list_y_encoded = []
        timer_update.tic()

        # load metadata
        exist_bit_arr = bitarray()
        exist_bit_arr.frombytes(zstd.decompress(ndb_utils.read_bytes_from_disk(os.path.join(comp_data_dir, 'exist_bit_arr.data'))))

        exp_data_dict = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'))
        num_record_per_part = exp_data_dict['num_record_per_part']
        data_ori_size = exp_data_dict['data_ori_size']
        data_comp_size = exp_data_dict['data_comp_size']
        max_len = exp_data_dict['max_len']
        x_start = exp_data_dict['x_start']
        x_end = exp_data_dict['x_end']
        list_y_encoder = exp_data_dict['list_y_encoder']
        num_tasks = exp_data_dict['num_tasks']
        size_encoder = exp_data_dict['size_encoder']

        if data_op == 'Insert' or data_op == 'Update':
            timer_encoding.tic()
            for col_idx in range(num_tasks):
                encoder = list_y_encoder[col_idx]
                # print(col_idx, data_change)
                list_y_encoded.append(encoder.transform(data_change.iloc[:, col_idx + 1]))
            x = data_change.iloc[:,0].astype(np.int32).values
            y = np.array(list_y_encoded).T.astype(np.int32)
            data = np.concatenate((x.reshape(-1,1), y), axis=1, dtype=np.int32)

            data_generator = DataGenerator(x, list_y_encoded, 1024*2**4, max_len, False, num_threads)
            exist_bit_arr_new = bitarray('0')*(np.max(x) - x_end)
            exist_bit_arr.extend(exist_bit_arr_new)
            x_end = x_end if x_end >= np.max(x) else np.max(x)
            # update bit array
            if data_op == 'Insert':
                # only needed for insert
                t_encoding += timer_encoding.toc()  
                        
                timer_nn.tic()
                for val in x:
                    exist_bit_arr[val-x_start] = 1

            

            misclassified_index = []
            correctclassified_index = []

            for idx, (x_sub,y_sub) in tqdm(enumerate(data_generator), total=len(data_generator)):
                y_sub = list(y_sub.values())
                if backend == 'tf':
                    y_sub_pred = model(x_sub)
                elif backend == 'onnx':
                    y_sub_pred = model.run(None, {input_name: np.expand_dims(x_sub, -1).astype(np.float32)})
                mis_pred = []

                for i in range(num_tasks):
                    if num_tasks == 1:
                        if backend == 'tf':
                            mis_pred.append(y_sub[i] != np.argmax(y_sub_pred, axis=1))
                        if backend == 'onnx':
                            mis_pred.append(y_sub[i] != np.argmax(y_sub_pred[0], axis=1))
                    else:
                        mis_pred.append(y_sub[i] != np.argmax(y_sub_pred[i], axis=1))
                
                misclassified = None

                for val in mis_pred:
                    if misclassified is None:
                            misclassified = val
                    else:
                        misclassified = np.logical_or(misclassified, val)
                        
                misclassified_index.extend(misclassified)

            correctclassified_index = np.array(misclassified_index) ^ 1
            misclassified_data = data[misclassified_index]
            correctclassified_data_index = data[correctclassified_index, 0]
            misclassified_data_comp = zstd.compress(misclassified_data.tobytes())
            print(sys.getsizeof(misclassified_data_comp)/1024/1024)

            t_nn += timer_nn.toc()
            timer_modify_aux.tic()
            # partition
            if len(misclassified_data) == 0:
                misclassified_data = np.zeros((1,2))
            record_size = misclassified_data[0].nbytes
            num_record_per_part = np.floor(block_size / record_size)

            x_start = np.min(misclassified_data[:,0])
            x_end = np.max(misclassified_data[:,0])
            x_range = x_end - x_start
            num_partition = int(math.ceil(x_range / num_record_per_part))
            # print(record_size*num_record_per_part/1024, num_partition)

            list_comp_aux_blocks = []
            comp_zstd_size = 0
            data_partition_idx = (misclassified_data[:, 0] - x_start) // num_record_per_part
            correct_pred_partition_idx = (correctclassified_data_index - x_start) // num_record_per_part

            data_partition_idx_start = int(x_start // num_record_per_part)

            num_threads_to_generate_file = max_generate_file_threads if max_generate_file_threads <= ndb_utils.get_sys_num_threads() else ndb_utils.get_sys_num_threads()
            num_partition_per_core = int(np.ceil(num_partition/num_threads_to_generate_file))
            arr_shm = shm.SharedMemory(create=True, size=misclassified_data.nbytes)
            shared_array = np.ndarray(misclassified_data.shape, dtype=misclassified_data.dtype, buffer=arr_shm.buf)
            shared_array[:] = misclassified_data[:]
            arr_shm_name = arr_shm.name
            arr_shm_dtypes = misclassified_data.dtype

            arr_shm1 = shm.SharedMemory(create=True, size=data_partition_idx.nbytes)
            shared_array1 = np.ndarray(data_partition_idx.shape, dtype=data_partition_idx.dtype, buffer=arr_shm1.buf)
            shared_array1[:] = data_partition_idx[:]
            arr_shm_name1 = arr_shm1.name
            arr_shm_dtypes1 = data_partition_idx.dtype

            arr_shm2 = shm.SharedMemory(create=True, size=correctclassified_data_index.nbytes)
            shared_array2 = np.ndarray(correctclassified_data_index.shape, dtype=correctclassified_data_index.dtype, buffer=arr_shm2.buf)
            shared_array2[:] = correctclassified_data_index[:]
            arr_shm_name2 = arr_shm2.name
            arr_shm_dtypes2 = correctclassified_data_index.dtype

            arr_shm3 = shm.SharedMemory(create=True, size=correct_pred_partition_idx.nbytes)
            shared_array3 = np.ndarray(correct_pred_partition_idx.shape, dtype=correct_pred_partition_idx.dtype, buffer=arr_shm3.buf)
            shared_array3[:] = correct_pred_partition_idx[:]
            arr_shm_name3 = arr_shm3.name
            arr_shm_dtypes3 = correct_pred_partition_idx.dtype

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads_to_generate_file) as executor:
                # Launch threads to process the numbers
                futures = [executor.submit(update_partition_to_disk, arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, arr_shm_name2, arr_shm_dtypes2,
                                        arr_shm_name3, arr_shm_dtypes3, misclassified_data.shape[0], misclassified_data.shape[1], data_partition_idx_start+i*num_partition_per_core, data_partition_idx_start+(i+1)*num_partition_per_core, comp_data_dir, zstd_compress_level, data_op) for i in range(num_threads_to_generate_file)]
                # Retrieve the results from each thread
                for future in concurrent.futures.as_completed(futures):
                    comp_zstd_size += future.result()

            arr_shm.close()
            arr_shm.unlink()
            arr_shm1.close()
            arr_shm1.unlink()
            arr_shm2.close()
            arr_shm2.unlink()
            arr_shm3.close()
            arr_shm3.unlink()
            t_modify_uax += timer_modify_aux.toc()

        elif data_op == 'Delete':
            x_range = x_end - x_start
            num_partition = int(math.ceil(x_range / num_record_per_part))
            data_idx_to_delete = data_change.values
            num_threads_to_generate_file = max_generate_file_threads if max_generate_file_threads <= ndb_utils.get_sys_num_threads() else ndb_utils.get_sys_num_threads()
            num_partition_per_core = int(np.ceil(num_partition/num_threads_to_generate_file))
            data_idx_to_delete_partition_idx = (data_idx_to_delete-x_start) // num_partition_per_core
            data_partition_idx_start = int(x_start // num_record_per_part)

            arr_shm2 = shm.SharedMemory(create=True, size=data_idx_to_delete.nbytes)
            shared_array2 = np.ndarray(data_idx_to_delete.shape, dtype=data_idx_to_delete.dtype, buffer=arr_shm2.buf)
            shared_array2[:] = data_idx_to_delete[:]
            arr_shm_name2 = arr_shm2.name
            arr_shm_dtypes2 = data_idx_to_delete.dtype

            arr_shm3 = shm.SharedMemory(create=True, size=data_idx_to_delete_partition_idx.nbytes)
            shared_array3 = np.ndarray(data_idx_to_delete_partition_idx.shape, dtype=data_idx_to_delete_partition_idx.dtype, buffer=arr_shm3.buf)
            shared_array3[:] = data_idx_to_delete_partition_idx[:]
            arr_shm_name3 = arr_shm3.name
            arr_shm_dtypes3 = data_idx_to_delete_partition_idx.dtype

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads_to_generate_file) as executor:
                # Launch threads to process the numbers
                futures = [executor.submit(delete_tuple_in_disk, arr_shm_name2, arr_shm_dtypes2,
                                        arr_shm_name3, arr_shm_dtypes3, data_idx_to_delete.shape[0], data_idx_to_delete.shape[1], data_partition_idx_start+i*num_partition_per_core, data_partition_idx_start+(i+1)*num_partition_per_core, comp_data_dir, zstd_compress_level, data_op) for i in range(num_threads_to_generate_file)]
                # Retrieve the results from each thread
                for future in concurrent.futures.as_completed(futures):
                    pass
            arr_shm2.close()
            arr_shm2.unlink()
            arr_shm3.close()
            arr_shm3.unlink()
            # update bit vector
            for val in data_idx_to_delete[:,0]:
                exist_bit_arr[val-x_start] = 1
        else:
            raise ValueError("Non-supported data op")
        
        # add save modified metadata here
        
        t_update += timer_update.toc()
    t_encoding /= num_loop
    t_nn /= num_loop
    t_modify_uax /= num_loop
    t_update /= num_loop
    print("info: time", t_encoding, t_nn, t_modify_uax, t_update)
    return t_update
