import pandas as pd 
import numpy as np
import sys
import zstd
import math
import os
from DeepMapping import ndb_utils
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow import keras
import ctypes
from sklearn import preprocessing
from bitarray import bitarray
from more_itertools import run_length
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
    def __init__(self, x, list_y, batch_size, max_len, shuffle=True):
        self.x = x
        self.list_y = list_y
        self.num_task = len(list_y)
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
    
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

        X = create_features_c_multi_thread(shared_utils, data_x, num_record, max_len)
        
        return X, Y
        
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

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
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    train_generator = DataGenerator(x, list_y_encoded, batch_size, max_len)

    if train == True:
        train_history = model.fit(train_generator, epochs=num_epochs, verbose=train_verbose, callbacks=[SOMT(model, 1)])
        return model, train_history
    else:
        return model, train_generator

def measure_latency_any(df, data_ori, task_name, sample_size, 
                    generate_file=True, memory_optimized=True, latency_optimized=True,
                    num_loop=10, num_query=5, search_algo='binary', path_to_model=None,
                    block_size=1024*1024):
    # TODO add support of hash to run-time memory optimized strategy
    # TODO add support of binary_c to run-time memory optimized strategy
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
        memory_optimized : bool
            whether measure the end-end latency with the run-time memory optimized strategy
        latency_optimized : bool
            whether measure the end-end latency with the latency optimized strategy
        num_loop : int
            number of loops to run for measuring the latency
        num_query : int
            number of queries
        search_algo : str
            search algorithm that applied to search entry in each partition
        path_to_model : str
            load model from custom path
    """
    data_ori_size = 0
    data_comp_size = 0
    memory_optimized_latency = None 
    latency_optimized_latency = None 
    memory_optimized_result = None
    latency_optimized_result = None

    df_key = [df.columns[0]]
    list_y_encoded = []
    list_y_encoder = []
    size_encoder = 0
    for col in df.columns:
        if col not in df_key:
            encoded_val, encoder = encode_label(df[col])
            list_y_encoded.append(encoded_val)
            list_y_encoder.append(encoder)
            size_encoder += encoder.classes_.nbytes
    num_tasks = len(list_y_encoded)
    
    for encoder in list_y_encoder:
        print(len(encoder.classes_))

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

    num_threads = 8

    x = df[df_key[0]].values.astype(np.int32)
    max_len = len(str(np.max(x)))
    y = np.array(list_y_encoded).T.astype(np.int32)
    data = np.concatenate((x.reshape(-1,1), y), axis=1, dtype=np.int32)
    print(data.nbytes/1024/1024)

    if path_to_model is None:
        model = tf.keras.models.load_model('models/nas/{}.h5'.format(task_name), compile=False)
    else:
        model = tf.keras.models.load_model(path_to_model, compile=False)
    train_generator = DataGenerator(x, list_y_encoded, 1024*2**4, max_len)

    # exist_bitarray
    x_start = np.min(x)
    x_end = np.max(x)
    exist_bit_arr = bitarray('0')*(x_end - x_start + 1)

    for val in x:
        exist_bit_arr[val-x_start] = 1
    print(sys.getsizeof(exist_bit_arr)/1024/1024)

    root_path = 'temp'
    folder_name = 'ours-any'
    comp_data_dir = os.path.join(root_path, task_name, folder_name)
    
    print('[Generate File Path]: {}'.format(comp_data_dir))
    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        misclassified_index = []

        for idx, (x_sub,y_sub) in tqdm(enumerate(train_generator), total=len(train_generator)):
            y_sub = list(y_sub.values())
            y_sub_pred = model(x_sub)
            mis_pred = []

            for i in range(num_tasks):
                if num_tasks == 1:
                    mis_pred.append(y_sub[i] != np.argmax(y_sub_pred, axis=1))
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
        # block_size = 1024 * 1024
        # block_size = 1024 * 512
        num_record_per_part = np.floor(block_size / record_size)

        x_start = np.min(misclassified_data[:,0])
        x_end = np.max(misclassified_data[:,0])
        x_range = x_end - x_start
        num_partition = int(math.ceil(x_range / num_record_per_part))
        print(record_size*num_record_per_part/1024, num_partition)

        list_comp_aux_blocks = []
        comp_zstd_size = 0
        for block_idx in tqdm(range(num_partition)):
            val_start, val_end = x_start + block_idx*num_record_per_part, x_start + (block_idx+1)*num_record_per_part
            data_idx = np.logical_and(misclassified_data[:, 0] >= val_start, misclassified_data[:, 0] < val_end)
            data_part = misclassified_data[data_idx]
            if len(data_part) == 0:
                continue
            data_bytes = data_part.tobytes()
            data_zstd_comp = zstd.compress(data_bytes,1)
            list_comp_aux_blocks.append(data_zstd_comp)
            comp_zstd_size += sys.getsizeof(data_zstd_comp)/1024/1024
            file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')
            ndb_utils.save_byte_to_disk(file_name, data_zstd_comp)

        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = [size_encoder, comp_zstd_size, model.count_params()*4/1024/1024, sys.getsizeof(zstd.compress(exist_bit_arr.tobytes()))/1024/1024]
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_comp_size))
        np.save(os.path.join(comp_data_dir, 'num_record_per_part'), num_record_per_part)
    else:
        num_record_per_part = np.load(os.path.join(comp_data_dir, 'num_record_per_part.npy'))  
    
    x = df[df_key[0]].values.astype(np.int32)
    max_len = len(str(np.max(x)))
    x_start = np.min(x)
    x_end = np.max(x)
    shared_utils.create_fetures.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int]
    shared_utils.create_fetures_mutlt_thread_mgr.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int32, ctypes.c_int32]
    shared_utils.create_fetures_mutlt_thread_mgr.restype = ctypes.POINTER(ctypes.c_bool * (sample_size * max_len * 10))
    list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
    # Measure latency for run-time memory optimzed strategy
    if memory_optimized:
        timer_creatfeatures = ndb_utils.Timer()
        timer_nn = ndb_utils.Timer()
        timer_aux_lookup = ndb_utils.Timer()
        timer_total = ndb_utils.Timer()
        timer_decomp = ndb_utils.Timer()
        timer_exist_lookup = ndb_utils.Timer()
        timer_sort = ndb_utils.Timer()
        timer_remap = ndb_utils.Timer()
        timer_locate_part = ndb_utils.Timer()
        t_remap = 0
        t_locate_part = 0
        t_decomp = 0
        t_createfeatures = 0
        t_aux_lookup = 0
        t_nn = 0
        t_exist_lookup = 0
        t_total = 0
        t_sort = 0
        peak_memory = -1
        block_bytes_size = 0

        timer_total.tic()
        for _ in tqdm(range(num_loop)):
            decomp_aux_block = None
            num_decomp = 0
            count_nonexist = 0
            prev_part_idx = None
            
            for query_idx in range(num_query):
                sample_index = list_sample_index[query_idx]              
                timer_total.tic()                                      
                timer_sort.tic()
                sample_index_sorted = np.sort(sample_index)
                sample_index_argsort = np.argsort(sample_index)
                t_sort += timer_sort.toc()               
                timer_creatfeatures.tic()               
                result = np.recarray((sample_size, ), dtype=data_ori.dtype)
                result[df_key[0]] = sample_index
                x_features_arr = np.zeros(sample_size * max_len * 10, dtype=bool)
                x_features_arr_ptr = shared_utils.create_fetures_mutlt_thread_mgr(
                    x_features_arr, sample_index, sample_size, max_len, num_threads)
                sampled_features = np.frombuffer(
                    x_features_arr_ptr.contents, dtype=bool).reshape(sample_size, -1)
                t_createfeatures += timer_creatfeatures.toc()
                # ---------
                timer_nn.tic()
                y_nn_pred = model(sampled_features)
                for i in range(num_tasks):
                    if num_tasks == 1:
                        col_name = data_ori.dtype.names[i+1]
                        result[col_name] = np.argmax(y_nn_pred, axis=1)
                    else:
                        col_name = data_ori.dtype.names[i+1]
                        result[col_name] = np.argmax(y_nn_pred[i], axis=1)
                t_nn += timer_nn.toc()
                for idx, val in enumerate(sample_index_sorted):
                    # ------ non exist look up
                    timer_exist_lookup.tic()
                    query_key = sample_index_sorted[idx]
                    query_key_index_in_old = sample_index_argsort[idx]
                    exist_flag = exist_bit_arr[query_key-x_start] == 1
                    t_exist_lookup += timer_exist_lookup.toc()
                    if not exist_flag:
                        result[idx] = -1
                        count_nonexist += 1
                        t_exist_lookup += timer_exist_lookup.toc()
                    else:
                        # misclassified lookup
                        t_exist_lookup += timer_exist_lookup.toc()
                        timer_locate_part.tic()
                        part_idx = int((query_key - x_start) // num_record_per_part)
                        t_locate_part += timer_locate_part.toc()
                        timer_decomp.tic()

                        if part_idx != prev_part_idx:
                            file_name = os.path.join(comp_data_dir, str(part_idx) + '.data')
                            if not os.path.exists(file_name):
                                continue
                            block_zstd_comp = ndb_utils.read_bytes_from_disk(file_name)
                            current_memory = sys.getsizeof(block_zstd_comp)
                            data_uncomp = np.frombuffer(zstd.decompress(block_zstd_comp), dtype=np.int32).reshape(-1, num_tasks+1).copy(order='F')

                            decomp_aux_block = data_uncomp
                            num_decomp += 1
                            current_memory += data_uncomp.nbytes
                            prev_part_idx = part_idx
                            if current_memory > peak_memory:
                                peak_memory = current_memory

                        else:
                            data_uncomp = decomp_aux_block
                        
                        t_decomp += timer_decomp.toc()                       
                        timer_aux_lookup.tic()
                        data_idx = shared_utils.aux_look_up_bin(data_uncomp[:,0], query_key, len(data_uncomp))
                        if data_idx != -1:
                            result[query_key_index_in_old] = tuple(data_uncomp[data_idx])
                        t_aux_lookup += timer_aux_lookup.toc()
                
                timer_remap.tic()
                for i in range(num_tasks):    
                    col_name = data_ori.dtype.names[i+1]
                    fun_a = lambda x: list_y_encoder[i].classes_[x]
                    result[col_name] = fun_a(result[col_name].astype(np.int32))
                t_remap += timer_remap.toc()
                t_total += timer_total.toc()


        peak_memory += exist_bit_arr.nbytes
        memory_optimized_result = result.copy()
        memory_optimized_latency = np.array((data_ori_size, np.sum(data_comp_size), sample_size, 0, peak_memory/1024/1024, t_sort / num_loop, t_createfeatures / num_loop, t_nn / num_loop, t_locate_part / num_loop, t_decomp / num_loop,
      t_aux_lookup / num_loop, t_exist_lookup / num_loop, t_remap / num_loop, t_total / num_loop, num_decomp, count_nonexist, exist_bit_arr.nbytes/1024/1024, model.count_params()*4/1024/1024)).T

    # Measure latency for end-end latency optimzed strategy
    if latency_optimized: 
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
            decomp_aux_block = dict()
            num_decomp = 0
            count_nonexist = 0
            peak_memory = 0
            cache_block_memory = 0

            # build hash table
            if search_algo == 'hash':
                data_hash = dict()
            for query_idx in range(num_query):
                sample_index = list_sample_index[query_idx]                
                timer_total.tic()                
                timer_sort.tic()
                sample_index_sorted = np.sort(sample_index)
                sample_index_argsort = np.argsort(sample_index)
                t_sort += timer_sort.toc()                              
                timer_creatfeatures.tic()        
                result = np.recarray((sample_size, ), dtype=data_ori.dtype)
                result[df_key[0]] = sample_index
                x_features_arr = np.zeros(sample_size * max_len * 10, dtype=bool)
                x_features_arr_ptr = shared_utils.create_fetures_mutlt_thread_mgr(
                    x_features_arr, sample_index, sample_size, max_len, num_threads)
                sampled_features = np.frombuffer(
                    x_features_arr_ptr.contents, dtype=bool).reshape(sample_size, -1)
                # sampled_features = ndb_utils.create_features(sample_index, max_len)[0]

                t_createfeatures += timer_creatfeatures.toc()
                timer_nn.tic()
                y_nn_pred = model(sampled_features)
                
                for i in range(num_tasks):
                    if num_tasks == 1:
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
                        part_idx = int((query_key - x_start) // num_record_per_part)
                        
                        t_locate_part += timer_locate_part.toc()
                        timer_decomp.tic()

                        if part_idx not in decomp_aux_block:
                            file_name = os.path.join(comp_data_dir, str(part_idx) + '.data')
                            if not os.path.exists(file_name):
                                continue
                            block_zstd_comp = ndb_utils.read_bytes_from_disk(file_name)
                            data_uncomp = np.frombuffer(
                            zstd.decompress(block_zstd_comp), dtype=np.int32).reshape(-1, num_tasks+1).copy(order='F')
                            decomp_aux_block[part_idx] = data_uncomp
                            num_decomp += 1
                            block_bytes_size = sys.getsizeof(block_zstd_comp)
                            prev_part_idx = part_idx

                            # TODO add size computation for hash approach
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
                        t_decomp += timer_decomp.toc()    
                        timer_aux_lookup.tic()
                        if search_algo == 'binary':
                            # TODO code can be optimized at revision stage
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

        peak_memory += exist_bit_arr.nbytes
        latency_optimized_result = result.copy()
        latency_optimized_latency = np.array((data_ori_size, np.sum(data_comp_size), sample_size, 1, peak_memory/1024/1024, t_sort / num_loop, t_createfeatures / num_loop, t_nn / num_loop, t_locate_part / num_loop, t_decomp / num_loop, t_build_index / num_loop,
      t_aux_lookup / num_loop, t_exist_lookup / num_loop, t_remap / num_loop, t_total / num_loop, num_decomp, count_nonexist, exist_bit_arr.nbytes/1024/1024, model.count_params()*4/1024/1024)).T

    return_latency = None 
    if memory_optimized_latency is None and latency_optimized_latency is not None:
        return_latency = latency_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is None:
        return_latency =  memory_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is not None:
        return_latency = np.vstack((memory_optimized_latency, latency_optimized_latency))

    return data_ori_size, data_comp_size, [memory_optimized_result, latency_optimized_result], return_latency