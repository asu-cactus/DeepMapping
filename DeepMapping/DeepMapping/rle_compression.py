import pandas as pd 
import numpy as np
import sys
import math
import os
from DeepMapping import ndb_utils
from DeepMapping.ndb_utils import Timer, recreate_temp_dir, save_byte_to_disk, read_bytes_from_disk
from more_itertools import run_length
from tqdm.auto import tqdm

def measure_latency(df, data_ori, task_name, sample_size, 
                    generate_file=True, memory_optimized=True, latency_optimized=True,
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
    folder_name = 'rle'
    comp_data_dir = os.path.join(root_path, task_name, folder_name)
    print('[Generate File Path]: {}'.format(comp_data_dir))

    list_type = []
    for col in data_ori.dtype.names:
        if data_ori[col].dtype == object:
            list_type.append({'names': [col], 'formats': ['O'], 'offsets': [0], 'itemsize': 8})
        elif data_ori[col].dtype == np.int32:
            list_type.append(np.int32)
        elif data_ori[col].dtype == np.float64:
            list_type.append(np.float64)

    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        data_size = 0
        count_na_part = 0
        list_rle_enabled = []
        for block_idx in tqdm(range(num_partition)):
            val_start, val_end = x_start + block_idx * \
                num_record_per_part, x_start + (block_idx+1)*num_record_per_part
            data_idx = np.logical_and(x >= val_start, x < val_end)
            if np.sum(data_idx) == 0 :
                continue
            
            data_part = data_ori[data_idx]
     
            for col_idx in range(df.shape[1]):
                col_name = df.columns[col_idx]
                col_val = data_ori[data_idx][col_name]
                if col_idx == 0:
                    data_bytes = col_val.tobytes()
                    file_name = os.path.join(comp_data_dir, str(block_idx) + '-{}.data'.format(col_idx))
                    ndb_utils.save_byte_to_disk(file_name, data_bytes)
                    data_size += sys.getsizeof(data_bytes)
                    list_rle_enabled.append(False)
                else:
                    col_val_rle_encode = list(run_length.encode(col_val))
                    if len(col_val_rle_encode)*2 > len(col_val):
                        # need more space when apply RLE
                        count_na_part += 1
                        data_bytes = col_val.tobytes()
                        file_name = os.path.join(comp_data_dir, str(block_idx) + '-{}.data'.format(col_idx))
                        ndb_utils.save_byte_to_disk(file_name, data_bytes)
                        data_size += sys.getsizeof(data_bytes)
                        list_rle_enabled.append(False)
                    else:
                        # col_rle_encode.append(col_val_rle_encode)
                        if list_type[col_idx] == np.int32:
                            temp_dtype = {'names': [data_ori.dtype.names[col_idx]], 'formats': [np.int32], 'offsets': [0], 'itemsize': 4}
                        else: 
                            temp_dtype = list_type[col_idx]
                        a = np.recarray((len(col_val_rle_encode),), dtype=temp_dtype)
                        b = np.zeros(len(col_val_rle_encode), np.int32)
                        for idx, val in enumerate(col_val_rle_encode):
                            a[idx] = val[0]
                            b[idx] = val[1]
                        file_name1 = os.path.join(comp_data_dir, str(block_idx) + '-{}-val.data'.format(col_idx))
                        file_name2 = os.path.join(comp_data_dir, str(block_idx) + '-{}-num.data'.format(col_idx))
                        ndb_utils.save_byte_to_disk(file_name1, a.tobytes())
                        ndb_utils.save_byte_to_disk(file_name2, b.tobytes())                    
                        data_size += a.nbytes + b.nbytes
                        list_rle_enabled.append(True)
        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = data_size/1024/1024            
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_size/1024/1024))
        np.save(os.path.join(comp_data_dir, 'list_rle_enabled'), list_rle_enabled)
    else:
        list_rle_enabled = np.load(os.path.join(comp_data_dir, 'list_rle_enabled.npy'))  
    
    list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)

    # Measure latency for run-time memory optimized strategy
    if memory_optimized:
        timer_total = ndb_utils.Timer()
        timer_decomp = ndb_utils.Timer()
        timer_sort = ndb_utils.Timer()
        timer_lookup = ndb_utils.Timer()
        timer_locate_part = ndb_utils.Timer() 
        t_total = 0
        t_decomp = 0
        t_lookup = 0
        t_sort = 0
        t_locate_part = 0
        peak_memory = 0

        for _ in tqdm(range(num_loop)):
            decomp_block = None
            num_decomp = 0
            count_nonexist = 0
            prev_part_idx = -1

            for query_idx in range(num_query):
                sample_index = list_sample_index[query_idx]
                timer_total.tic()
                timer_sort.tic()
                sample_index_sorted = np.sort(sample_index)
                sample_index_argsort = np.argsort(sample_index)
                t_sort += timer_sort.toc()
                result = np.recarray((sample_size,), dtype=data_ori.dtype)

                for idx in range(sample_size):
                    timer_locate_part.tic() 
                    query_key = sample_index_sorted[idx]
                    query_key_index_in_old = sample_index_argsort[idx]
                    part_idx = int((query_key-x_start) // num_record_per_part)
                    t_locate_part += timer_locate_part.toc()
                    timer_decomp.tic()
                    if part_idx != prev_part_idx:     
                        current_memory = 0   
                        # decompress index first
                        file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(0))
                        block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                        block_data = np.frombuffer(block_bytes, dtype=list_type[0]) 
                        curr_decomp_block = np.recarray((len(block_data),), dtype=data_ori.dtype)                        
                        current_memory += sys.getsizeof(block_bytes)                       
                        curr_decomp_block[curr_decomp_block.dtype.names[0]] = block_data
                        
                        for i in range(1, df.shape[1]):
                            col_name = data_ori.dtype.names[i]

                            if list_rle_enabled[i + part_idx*df.shape[1]] == False:
                                file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(i))
                                block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                                current_memory += sys.getsizeof(block_bytes)

                                if list_type[i] == np.int32 or list_type[i] == np.float64:
                                    block_data = np.frombuffer(block_bytes, dtype=list_type[i])
                                else:
                                    block_data = np.rec.array(block_bytes, dtype=list_type[i])[col_name]
                                curr_decomp_block[col_name] = block_data
                            else:
                                # rle decode
                                if list_type[i] == np.int32:
                                    temp_dtype = {'names': [data_ori.dtype.names[i]], 'formats': [np.int32], 'offsets': [0], 'itemsize': 4}
                                else: 
                                    temp_dtype = list_type[i]
                            
                                file_name1 = os.path.join(comp_data_dir, str(part_idx) + '-{}-val.data'.format(i))
                                file_name2 = os.path.join(comp_data_dir, str(part_idx) + '-{}-num.data'.format(i))
                                val_data = np.rec.array(ndb_utils.read_bytes_from_disk(file_name1), dtype=temp_dtype)
                                num_data = np.frombuffer(ndb_utils.read_bytes_from_disk(file_name2), dtype=np.int32)
                                temp_col_decode_data = []

                                for val, num in zip(val_data, num_data):
                                    temp_col_decode_data.extend([val[0]]*num)
                                curr_decomp_block[col_name] = temp_col_decode_data
                                
                                current_memory += val_data.nbytes
                                current_memory += num_data.nbytes
          
                        current_memory += curr_decomp_block.nbytes
                        
                        decomp_block = curr_decomp_block
                        num_decomp += 1                
                        if current_memory > peak_memory:
                            peak_memory = current_memory
                        prev_part_idx = part_idx
                    else:
                        curr_decomp_block = decomp_block

                    t_decomp += timer_decomp.toc()
                    timer_lookup.tic()

                    if search_algo == 'binary':
                        data_idx = ndb_utils.binary_search(curr_decomp_block[key], query_key, len(curr_decomp_block))
                    elif search_algo == 'naive':
                        data_idx = curr_decomp_block[key] == query_key

                    if (search_algo == 'binary' and data_idx >= 0) or (search_algo == 'naive' and np.sum(data_idx) > 0):
                        result[query_key_index_in_old] = curr_decomp_block[data_idx]
                    else:
                        count_nonexist += 1
                    
                    t_lookup += timer_lookup.toc()

                t_total += timer_total.toc()
        memory_optimized_result = result.copy()
        memory_optimized_latency = np.array((data_ori_size, data_comp_size, sample_size, 0, peak_memory/1024/1024, t_sort / num_loop, 
        t_locate_part / num_loop, t_decomp / num_loop, 
        t_lookup / num_loop, t_total / num_loop, num_decomp, count_nonexist)).T

    # Measure latency for end-end latency optimzed strategy
    if latency_optimized: 
        timer_total = ndb_utils.Timer()
        timer_decomp = ndb_utils.Timer()
        timer_lookup = ndb_utils.Timer()
        timer_total = ndb_utils.Timer()
        timer_sort = ndb_utils.Timer()
        timer_locate_part = ndb_utils.Timer()
        t_decomp = 0
        t_lookup = 0
        t_sort = 0
        t_total = 0
        t_locate_part = 0
        timer_total.tic()

        for _ in tqdm(range(num_loop)):  
            decomp_block = dict()
            peak_memory = 0
            num_decomp = 0
            count_nonexist = 0
            cache_block_memory = 0

            for query_idx in range(num_query):
                sample_index = list_sample_index[query_idx]
                timer_total.tic()
                timer_sort.tic()
                sample_index_sorted = np.sort(sample_index)
                sample_index_argsort = np.argsort(sample_index)
                t_sort += timer_sort.toc()
                result = np.recarray((sample_size,), dtype=data_ori.dtype)
                result_idx = 0
                
                for idx in range(sample_size):
                    timer_locate_part.tic()        
                    query_key = sample_index_sorted[idx]
                    query_key_index_in_old = sample_index_argsort[idx]            
                    t_locate_part += timer_locate_part.toc()

                    part_idx = int((query_key-x_start) // num_record_per_part)
                    timer_decomp.tic()
                    decomp_memory = 0
                    if part_idx not in decomp_block:
                        # decompress index first
                        file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(0))
                        block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                        block_data = np.frombuffer(block_bytes, dtype=list_type[0])
                        curr_decomp_block = np.recarray((len(block_data),), dtype=data_ori.dtype)   
                        decomp_memory += sys.getsizeof(block_bytes)   
                        curr_decomp_block[curr_decomp_block.dtype.names[0]] = block_data
                        
                        for i in range(1, df.shape[1]):
                            col_name = data_ori.dtype.names[i]
                            if list_rle_enabled[i + part_idx*df.shape[1]] == False:
                                file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(i))
                                block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                                if list_type[i] == np.int32 or list_type[i] == np.float64:
                                    block_data = np.frombuffer(block_bytes, dtype=list_type[i])
                                else:
                                    block_data = np.rec.array(block_bytes, dtype=list_type[i])[col_name]
                                curr_decomp_block[col_name] = block_data
                                decomp_memory += sys.getsizeof(block_bytes)
                            else:
                                # rle decode
                                if list_type[i] == np.int32:
                                    temp_dtype = {'names': [data_ori.dtype.names[i]], 'formats': [np.int32], 'offsets': [0], 'itemsize': 4}
                                else: 
                                    temp_dtype = list_type[i]
                                    
                                file_name1 = os.path.join(comp_data_dir, str(part_idx) + '-{}-val.data'.format(i))
                                file_name2 = os.path.join(comp_data_dir, str(part_idx) + '-{}-num.data'.format(i))
                                val_data = np.rec.array(ndb_utils.read_bytes_from_disk(file_name1), dtype=temp_dtype)
                                num_data = np.frombuffer(ndb_utils.read_bytes_from_disk(file_name2), dtype=np.int32)
                                temp_col_decode_data = []

                                for val, num in zip(val_data, num_data):
                                    temp_col_decode_data.extend([val[0]]*num)
                                curr_decomp_block[col_name] = temp_col_decode_data
                                decomp_memory += val_data.nbytes
                                decomp_memory += num_data.nbytes
                            
                        cache_block_memory += curr_decomp_block.nbytes  
                        decomp_block[part_idx] = curr_decomp_block
                        num_decomp += 1
                    else:
                        curr_decomp_block = decomp_block[part_idx]
                    t_decomp += timer_decomp.toc()
                    # -----
                    timer_lookup.tic()

                    if search_algo == 'binary':
                        data_idx = ndb_utils.binary_search(curr_decomp_block[key], query_key, len(curr_decomp_block))
                    elif search_algo == 'naive':
                        data_idx = curr_decomp_block[key] == query_key

                    if (search_algo == 'binary' and data_idx >= 0) or (search_algo == 'naive' and np.sum(data_idx) > 0):
                        result[query_key_index_in_old] = curr_decomp_block[data_idx]
                    else:
                        count_nonexist += 1
                    t_lookup += timer_lookup.toc()
                    result_idx += 1
                    if cache_block_memory + decomp_memory > peak_memory:
                        peak_memory = cache_block_memory + decomp_memory
                t_total += timer_total.toc()
        latency_optimized_result = result.copy()
        latency_optimized_latency = np.array((data_ori_size, data_comp_size, sample_size, 1, peak_memory/1024/1024, t_sort / num_loop, 
        t_locate_part / num_loop, t_decomp / num_loop, 
        t_lookup / num_loop, t_total / num_loop, num_decomp, count_nonexist)).T

    return_latency = None 
    if memory_optimized_latency is None and latency_optimized_latency is not None:
        return_latency = latency_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is None:
        return_latency =  memory_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is not None:
        return_latency = np.vstack((memory_optimized_latency, latency_optimized_latency))

    return data_ori_size, data_comp_size, [memory_optimized_result, latency_optimized_result], return_latency