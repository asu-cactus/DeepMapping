import pandas as pd 
import numpy as np
import sys
import math
import os
import ctypes
from DeepMapping import ndb_utils
# from DeepMapping.ndb_utils import Timer, recreate_temp_dir, save_byte_to_disk, read_bytes_from_disk
# from more_itertools import run_length
from tqdm.auto import tqdm

ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.bool_, 
                                            ndim=1,
                                            flags="C")
ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, 
                                    ndim=1,
                                    flags="C")

shared_utils = ctypes.CDLL(os.path.abspath("shared_utils.so")) # Or full path to file 
shared_utils.aux_look_up_bin.argtypes = [ND_POINTER_2, ctypes.c_int, ctypes.c_long]
shared_utils.aux_look_up_bin.restype = ctypes.c_long

def measure_latency(df, data_ori, task_name, sample_size, 
                    generate_file=True, memory_optimized=True, latency_optimized=True,
                    num_loop=10, num_query=5, search_algo='binary', block_size=1024*1024):
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
            search algorithm that applied to search entry in each partition, available strategy: naive, binary, hash
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
    # block_size = 1024 * 1024
    # block_size = 1024 * 512
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
    folder_name = 'uncompress'
    comp_data_dir = os.path.join(root_path, task_name, folder_name)
    print('[Generate File Path]: {}'.format(comp_data_dir))

    dict_contigous_key = dict()
    
    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        data_size = 0

        for block_idx in tqdm(range(num_partition)):
            val_start, val_end = x_start + block_idx * \
                num_record_per_part, x_start + (block_idx+1)*num_record_per_part
            data_idx = np.logical_and(x >= val_start, x < val_end)
            data_part = data_ori[data_idx]
            if search_algo == 'binary_c':
                # FIXME temporary workaround to avoid the overhead of converting to contiguous array
                dict_contigous_key[block_idx] = np.array(data_part[key], order='F').astype(np.int32)

            if len(data_part) == 0:
                continue
            data_bytes = data_part.tobytes()
            data_size += sys.getsizeof(data_bytes)
            file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')
            ndb_utils.save_byte_to_disk(file_name, data_bytes)

        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = data_size/1024/1024
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_size/1024/1024))
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
                        # new block to decompress
                        file_name = os.path.join(comp_data_dir, str(part_idx) + '.data')
                        block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                        curr_decomp_block = np.rec.array(block_bytes, dtype=data_ori.dtype)
                        decomp_block = curr_decomp_block
                        num_decomp += 1
                        current_memory = sys.getsizeof(block_bytes)
                        current_memory += curr_decomp_block.nbytes
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
        timer_build_index = ndb_utils.Timer()
        timer_locate_part = ndb_utils.Timer()
        t_decomp = 0
        t_lookup = 0
        t_sort = 0
        t_total = 0
        t_locate_part = 0
        t_build_index = 0
        timer_total.tic()
        for _ in tqdm(range(num_loop)):  
            decomp_block = dict()
            peak_memory = 0
            num_decomp = 0
            count_nonexist = 0
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
                result = np.recarray((sample_size,), dtype=data_ori.dtype)
                result_idx = 0
                
                for idx in range(sample_size):
                    timer_locate_part.tic()        
                    query_key = sample_index_sorted[idx]
                    query_key_index_in_old = sample_index_argsort[idx]
                    t_locate_part += timer_locate_part.toc()
                    part_idx = int((query_key-x_start) // num_record_per_part)
                    timer_decomp.tic()

                    if part_idx not in decomp_block:
                        file_name = os.path.join(comp_data_dir, str(part_idx) + '.data')
                        block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                        curr_decomp_block = np.rec.array(block_bytes, dtype=data_ori.dtype)
                        decomp_block[part_idx] = curr_decomp_block
                        num_decomp += 1
                        block_bytes_size = sys.getsizeof(block_bytes)

                        if search_algo == 'hash':
                            t_decomp += timer_decomp.toc()
                            timer_build_index.tic()
                            for block_data_idx in range(len(curr_decomp_block)):
                                data_entry_key = curr_decomp_block[key][block_data_idx]
                                # print(data_entry_key)
                                data_entry_val = curr_decomp_block[block_data_idx]
                                data_hash[data_entry_key] = data_entry_val   
                            cache_block_memory = sys.getsizeof(data_hash)
                            t_build_index += timer_build_index.toc()
                            timer_decomp.tic()
                        else:
                            cache_block_memory += curr_decomp_block.nbytes 
                    else:
                        curr_decomp_block = decomp_block[part_idx]

                    t_decomp += timer_decomp.toc()
                    timer_lookup.tic()

                    if search_algo == 'binary':
                        data_idx = ndb_utils.binary_search(curr_decomp_block[key], query_key, len(curr_decomp_block))
                    elif search_algo == 'binary_c':
                        arr_contigous_arr = dict_contigous_key[part_idx]
                        data_idx = shared_utils.aux_look_up_bin(arr_contigous_arr, query_key, len(curr_decomp_block))
                    elif search_algo == 'naive':
                        data_idx = curr_decomp_block[key] == query_key
                    elif search_algo == 'hash':
                        data_idx = query_key in data_hash.keys()                            

                    if ((search_algo == 'binary' or search_algo =='binary_c') and data_idx >= 0) or (search_algo == 'naive' and np.sum(data_idx) > 0):
                        result[query_key_index_in_old] = curr_decomp_block[data_idx]
                    elif search_algo == 'hash' and data_idx == True:
                        result[query_key_index_in_old] = data_hash[query_key]
                    else:
                        count_nonexist += 1

                    t_lookup += timer_lookup.toc()
                    result_idx += 1
                    if cache_block_memory + block_bytes_size > peak_memory:
                        peak_memory = cache_block_memory + block_bytes_size
                t_total += timer_total.toc()
        latency_optimized_result = result.copy()
        latency_optimized_latency = np.array((data_ori_size, data_comp_size, sample_size, 1, peak_memory/1024/1024, t_sort / num_loop, 
        t_locate_part / num_loop, t_decomp / num_loop, t_build_index / num_loop,
        t_lookup / num_loop, t_total / num_loop, num_decomp, count_nonexist)).T

    return_latency = None 
    if memory_optimized_latency is None and latency_optimized_latency is not None:
        return_latency = latency_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is None:
        return_latency =  memory_optimized_latency.reshape((1,-1))
    elif memory_optimized_latency is not None and latency_optimized_latency is not None:
        return_latency = np.vstack((memory_optimized_latency, latency_optimized_latency))

    return data_ori_size, data_comp_size, [memory_optimized_result, latency_optimized_result], return_latency
