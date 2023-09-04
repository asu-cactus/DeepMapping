import gc
import math
import numpy as np
import os
import pandas as pd 
import sys
from DeepMapping import ndb_utils
from sklearn import preprocessing
from DeepMapping.ndb_utils import Timer, recreate_temp_dir, save_byte_to_disk, read_bytes_from_disk
from tqdm.auto import tqdm

def encode_label(arr):
    """encode the label
    """
    label_encoder = preprocessing.LabelEncoder().fit(arr)
    arr_encode = label_encoder.transform(arr)
    return arr_encode, label_encoder

def min_encode_bit(arr):
    # obtain the minimun bits required for encoding
    # check paper Efﬁcient Query Processing with Optimistically Compressed Hash Tables & Strings in the USSR Figure 2
    # return minimun value of the array, maximum value of the array, minimun bits required to encode the arr
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    min_bit = int(np.ceil(np.log2(arr_max - arr_min + 1)))
    return arr_min, arr_max, min_bit

def encode_with_min_bits(arr, required_bits):
    dtype = None
    if required_bits == 8:
        dtype = np.uint8
    elif required_bits == 16:
        dtype = np.uint16
    elif required_bits == 32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    return arr.astype(dtype), dtype

def dict_compression(df):
    dict_comp_data = []
    list_encoder = []
    for col in tqdm(df.columns, total=df.shape[1]):
        vals = df[col]
        if pd.api.types.is_integer_dtype(vals):
            dict_comp_data.append(vals.values)
            list_encoder.append(None)
        else:
            encoded_val, encoder = encode_label(vals.values)
            print(len(encoder.classes_))
            required_bits = int(np.ceil(np.log2(len(encoder.classes_))/8)*8)  
            encoded_val_recode, dtype = encode_with_min_bits(encoded_val, required_bits)
            print('COL: {} encode use: {}'.format(col, dtype))
            dict_comp_data.append(encoded_val_recode)
            list_encoder.append(encoder)
    return dict_comp_data, list_encoder

def measure_latency(df, data_ori, task_name, sample_size, 
                    generate_file=True,
                    num_loop=10, num_query=5, search_algo='binary'):
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
    """
    mode = os.environ['MODE']
    data_ori_size = 0
    data_comp_size = 0
    memory_optimized_latency = None 
    latency_optimized_latency = None 
    memory_optimized_result = None
    latency_optimized_result = None
    exp_data_dict = dict()
    dict_comp_data, dict_encoder = dict_compression(df)
    list_type = []

    for arr in dict_comp_data:
        list_type.append(arr.dtype)

    key = df.columns[0]
    block_size = 1024 * 1024
    record_size = data_ori[0].nbytes
    num_record_per_part = np.floor(block_size / record_size)
    x = data_ori[key]
    x_start = np.min(x)
    x_end = np.max(x)
    x_range = x_end - x_start
    num_partition = int(math.ceil(x_range / num_record_per_part))
    print('[Partition] Size {} Per Partition, # Partition: {}'.format(record_size*num_record_per_part/1024, num_partition))
    root_path = 'temp'
    task_name = task_name
    folder_name = 'dict'
    comp_data_dir = os.path.join(root_path, task_name, folder_name)
    print('[Generate File Path]: {}'.format(comp_data_dir))

    # generate file
    if generate_file:
        recreate_temp_dir(comp_data_dir)
        data_size = 0

        for block_idx in tqdm(range(num_partition)):
            val_start, val_end = x_start + block_idx * \
                num_record_per_part, x_start + (block_idx+1)*num_record_per_part
            data_idx = np.logical_and(x >= val_start, x < val_end)

            if np.sum(data_idx) == 0 :
                continue
            
            for i in range(len(dict_comp_data)):
                data_part = dict_comp_data[i][data_idx]
                data_part_bytes = data_part.tobytes()
                data_size += sys.getsizeof(data_part_bytes)
                file_name = os.path.join(comp_data_dir, str(block_idx) + '-{}.data'.format(i))
                save_byte_to_disk(file_name, data_part_bytes)
                
        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = data_size/1024/1024
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_size/1024/1024))
        exp_data_dict['num_record_per_part'] = num_record_per_part 
        exp_data_dict['data_ori_size'] = data_ori_size
        exp_data_dict['data_comp_size'] = data_comp_size
        exp_data_dict['x_start'] = x_start 
        exp_data_dict['x_end'] = x_end 
        exp_data_dict['list_type'] = list_type
        exp_data_dict['dict_compressor'] = dict_comp_data, dict_encoder
        ndb_utils.save_obj_to_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'), exp_data_dict)
        list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
    else:
        exp_data_dict = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'))
        num_record_per_part = exp_data_dict['num_record_per_part']
        data_ori_size = exp_data_dict['data_ori_size']
        data_comp_size = exp_data_dict['data_comp_size']
        x_start = exp_data_dict['x_start']
        x_end = exp_data_dict['x_end']
        list_type = exp_data_dict['list_type']
        dict_comp_data, dict_encoder = exp_data_dict['dict_compressor']
        list_sample_index = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(root_path, task_name, 'sample_index_{}.data'.format(sample_size)))
    
    list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)

    # Measure latency for run-time memory optimized strategy
   
    timer_total = Timer()
    timer_decomp = Timer()
    timer_lookup = Timer()
    timer_total = Timer()
    timer_sort = Timer()
    timer_locate_part = Timer()
    t_decomp = 0
    t_lookup = 0
    t_sort = 0
    t_total = 0
    t_locate_part = 0
    timer_total.tic()

    for _ in tqdm(range(num_loop)):  
        decomp_block = dict()
        partition_hit = dict()
        peak_memory = 0
        num_decomp = 0
        count_nonexist = 0
        cache_block_memory = 0
        gc.collect()

        for query_idx in range(num_query):
            sample_index = list_sample_index[query_idx]
            timer_total.tic()
            timer_sort.tic()
            sample_index_sorted = np.sort(sample_index)
            sample_index_argsort = np.argsort(sample_index)
            t_sort += timer_sort.toc()
            result = np.ndarray((sample_size,), dtype=data_ori.dtype)
            result_idx = 0

            for idx in range(sample_size):
                timer_locate_part.tic()
                query_key = sample_index_sorted[idx]
                query_key_index_in_old = sample_index_argsort[idx]
                t_locate_part += timer_locate_part.toc()

                part_idx = int((query_key-x_start) // num_record_per_part)
                timer_decomp.tic()
                # -----
                decomp_memory = 0
                if part_idx not in decomp_block:
                    if mode == 'edge':
                        available_memory = ndb_utils.get_available_memory()
                        if available_memory < 1024*1024*100:
                            # memory not eneough, free some memory
                            decomp_block = ndb_utils.evict_unused_partition(decomp_block, partition_hit, free_memory=1024*1024*100)

                    partition_hit[part_idx] =1

                    # decompress index first
                    file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(0))
                    block_bytes = read_bytes_from_disk(file_name)
                    block_data = np.frombuffer(block_bytes, dtype=list_type[0])
                    curr_decomp_block = np.ndarray((len(block_data),), dtype=data_ori.dtype)                       
                    decomp_memory += sys.getsizeof(block_bytes)                       
                    curr_decomp_block[curr_decomp_block.dtype.names[0]] = block_data
                    
                    for i in range(1, len(dict_comp_data)):
                        file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(i))
                        block_bytes = read_bytes_from_disk(file_name)
                        decomp_memory += sys.getsizeof(block_bytes)
                        block_data = np.frombuffer(block_bytes, dtype=list_type[i])
                        col_name = data_ori.dtype.names[i]

                        if dict_encoder[i] is not None:
                            # encoded col                                
                            fun_a = lambda x: dict_encoder[i].classes_[x]
                            curr_decomp_block[col_name] = fun_a(block_data.astype(np.int32))                               
                        else:
                            curr_decomp_block[col_name] = block_data
                        
                    cache_block_memory += curr_decomp_block.nbytes
                    decomp_block[part_idx] = curr_decomp_block
                    num_decomp += 1
                else:
                    partition_hit[part_idx] += 1
                    curr_decomp_block = decomp_block[part_idx]
                t_decomp += timer_decomp.toc()
                timer_lookup.tic()

                if search_algo == 'binary':
                    data_idx = ndb_utils.binary_search(curr_decomp_block[key], query_key, len(curr_decomp_block))
                elif search_algo == 'naive':
                    data_idx = curr_decomp_block[key] == query_key

                if (search_algo == 'binary' and data_idx >= 0) or (search_algo == 'naive' and np.sum(data_idx) > 0):
                    result[query_key_index_in_old] = tuple(curr_decomp_block[data_idx])
                else:
                    count_nonexist += 1

                t_lookup += timer_lookup.toc()
                result_idx += 1

                if cache_block_memory + decomp_memory > peak_memory:
                    peak_memory = cache_block_memory + decomp_memory

            t_total += timer_total.toc()
    latency_optimized_result = result.copy()
    latency_optimized_latency = np.array((data_ori_size, data_comp_size, sample_size, 1, peak_memory/1024/1024, t_sort / num_loop, 
    t_locate_part / num_loop, t_decomp / num_loop,  0 / num_loop, # build_index time #TODO this is required for build hash table, current is no needed, use binary search instead
    t_lookup / num_loop, t_total / num_loop, num_decomp, count_nonexist)).T

    return_latency = None 

    if memory_optimized_latency is None and latency_optimized_latency is not None:
        return_latency = latency_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is None:
        return_latency =  memory_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is not None:
        return_latency = np.vstack((memory_optimized_latency, latency_optimized_latency))

    return data_ori_size, data_comp_size, [memory_optimized_result, latency_optimized_result], return_latency