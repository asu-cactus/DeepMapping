import ctypes
import gc
import json
import math
import numpy as np
import os
import pandas as pd 
import pickle
import sys
import zstd
from DeepMapping import ndb_utils
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
                    generate_file=True,
                    num_loop=10, num_query=5, search_algo='binary', block_size=1024*1024):
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
            search algorithm that applied to search entry in each partition, available strategy: naive, binary, hash
        path_to_model : str
            load model from custom path
    """
    mode = os.environ['MODE']
    data_ori_size = 0
    data_comp_size = 0
    memory_optimized_latency = None 
    latency_optimized_latency = None 
    memory_optimized_result = None
    latency_optimized_result = None
    exp_data_dict = dict()

    key = df.columns[0]
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
    folder_name = 'hashtable_with_compression'
    comp_data_dir = os.path.join(root_path, task_name, folder_name)
    if 'DATA_OPS' in os.environ:
        comp_data_dir = os.path.join(comp_data_dir, os.environ['DATA_OPS'])
        
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
                dict_contigous_key[block_idx] = np.array(data_part[key], order='F').astype(np.int32)

            if len(data_part) == 0:
                continue
            data_part_hash_table = dict()
            for data_idx in range(len(data_part)):
                data_part_hash_table[data_part[key][data_idx]] = data_part[data_idx]
            
            data_part_hash_table_bytes = pickle.dumps(data_part_hash_table)
            # data_size += sys.getsizeof(data_bytes)
            file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')

            # data_part_hash_table_bytes = zstd.compress(json.dumps(data_part_hash_table).encode('utf-8'))
            # ndb_utils.save_hashtable_to_disk(file_name, data_part_hash_table)
            ndb_utils.save_byte_to_disk(file_name, zstd.compress(data_part_hash_table_bytes))
            data_size += os.path.getsize(file_name)

        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = data_size/1024/1024
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_size/1024/1024))
        exp_data_dict['num_record_per_part'] = num_record_per_part 
        exp_data_dict['data_ori_size'] = data_ori_size
        exp_data_dict['data_comp_size'] = data_comp_size
        exp_data_dict['x_start'] = x_start 
        exp_data_dict['x_end'] = x_end 
        ndb_utils.save_obj_to_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'), exp_data_dict)
        list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
    else:
        exp_data_dict = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'))
        num_record_per_part = exp_data_dict['num_record_per_part']
        data_ori_size = exp_data_dict['data_ori_size']
        data_comp_size = exp_data_dict['data_comp_size']
        x_start = exp_data_dict['x_start']
        x_end = exp_data_dict['x_end']
        list_sample_index = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(root_path, task_name, 'sample_index_{}.data'.format(sample_size)))

  
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
        partition_hit = dict() 
        decomp_block = dict()
        peak_memory = 0
        num_decomp = 0
        count_nonexist = 0
        cache_block_memory = 0
        gc.collect()

        # build hash table
        # if search_algo == 'hash':
        #     data_hash = dict() 

        for query_idx in range(num_query):
            sample_index = list_sample_index[query_idx]
            timer_total.tic()
            timer_sort.tic()
            sample_index_sorted = np.sort(sample_index)
            sample_index_argsort = np.argsort(sample_index)
            sample_index_partition = (sample_index_sorted - x_start) // num_record_per_part
            sample_index_partition = sample_index_partition.astype(np.int32)
            t_sort += timer_sort.toc()
            result = np.recarray((sample_size,), dtype=data_ori.dtype)
            result_idx = 0
            
            for idx in range(sample_size):
                query_key = sample_index_sorted[idx]
                query_key_index_in_old = sample_index_argsort[idx]
                timer_locate_part.tic()        
                # part_idx = int((query_key-x_start) // num_record_per_part)
                part_idx = sample_index_partition[idx]
                t_locate_part += timer_locate_part.toc()
                timer_decomp.tic()

                if part_idx not in decomp_block:
                    if mode == 'edge':
                        available_memory = ndb_utils.get_available_memory()
                        if available_memory < 1024*1024*100:
                            # memory not eneough, free some memory
                            decomp_block = ndb_utils.evict_unused_partition(decomp_block, partition_hit, free_memory=1024*1024*100)
                    partition_hit[part_idx] = 1
                    file_name = os.path.join(comp_data_dir, str(part_idx) + '.data')
                    block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                    curr_decomp_block = pickle.loads(zstd.uncompress(block_bytes))
                    # curr_decomp_block = ndb_utils.load_hashtable_from_disk(file_name)
                    try:
                        decomp_block[part_idx] = curr_decomp_block
                    except:
                        decomp_block = dict()
                        decomp_block[part_idx] = curr_decomp_block
                    num_decomp += 1
                    load_block_bytes = sys.getsizeof(block_bytes)
                    block_bytes_size = ndb_utils.get_nested_dict_size(curr_decomp_block)

                    cache_block_memory += block_bytes_size

                    if search_algo == 'hash':
                        t_decomp += timer_decomp.toc()
                        timer_build_index.tic()
                        t_build_index += timer_build_index.toc()
                        timer_decomp.tic()
                    else:
                        pass
                else:
                    curr_decomp_block = decomp_block[part_idx]
                    partition_hit[part_idx] += 1

                t_decomp += timer_decomp.toc()
                timer_lookup.tic()

                data_idx = query_key in curr_decomp_block.keys()                            
                if data_idx == True:
                    result[query_key_index_in_old] = tuple(curr_decomp_block[query_key])
                else: 
                    count_nonexist += 1

                t_lookup += timer_lookup.toc()
                result_idx += 1
                if cache_block_memory + load_block_bytes > peak_memory:
                    peak_memory = cache_block_memory + load_block_bytes
            t_total += timer_total.toc()
        latency_optimized_result = result.copy()
        del result
        gc.collect()
        print('[DEBUG] number of decompressed partition', len(decomp_block))
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
