import gc
import numpy as np
import math
import os
import pandas as pd 
import sys
import warnings
from DeepMapping import ndb_utils
from tqdm.auto import tqdm



def measure_latency(df, data_ori, task_name, sample_size, 
                    generate_file=True,
                    num_loop=10, num_query=5, search_algo='binary', block_size=1024*1024, **kwargs):
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
        block_size  : int
            block size for each partition, size in bytes
    """
    mode = os.environ['MODE']
    data_ori_size = 0
    data_comp_size = 0
    memory_optimized_latency = None 
    arr_latency = None 
    arr_result = None
    exp_data_dict = dict()

    list_type = []
    for col in data_ori.dtype.names:
        if data_ori[col].dtype == 'S8':
            list_type.append((col, 'S8'))
        elif data_ori[col].dtype == np.float64:
            list_type.append(np.float64)
        else:
            list_type.append(np.int32)

    root_path = 'temp'
    if os.environ['MODE'] == 'tuning':
        root_path = 'temp_tune'
    folder_name = 'delta'
    comp_data_dir = os.path.join(root_path, task_name, folder_name, str(block_size))
    print('[Generate File Path]: {}'.format(comp_data_dir))

    key = df.columns[0]
    record_size = data_ori[0].nbytes
    num_record_per_part = np.floor(block_size / record_size)
    x = data_ori[key]
    x_start = np.min(x)
    x_end = np.max(x)
    x_range = x_end - x_start
    num_partition = int(math.ceil(x_range / num_record_per_part))
    print('[Partition] Size {} Per Partition, # Partition: {}'.format(record_size*num_record_per_part/1024, num_partition))
    diff_dtype = np.int8

    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        data_size = 0     
        data_partition_idx = (x - x_start) // num_record_per_part
        for block_idx in tqdm(range(num_partition)):
            data_idx = data_partition_idx == block_idx
            if np.sum(data_idx) == 0 :
                continue           
            data_part = data_ori[data_idx]   

            for col_idx in range(df.shape[1]):
                list_delta_enabled = []
                col_name = df.columns[col_idx]
                col_val = data_ori[data_idx][col_name]
                if col_idx == 0:
                    data_bytes = col_val.tobytes()
                    file_name = os.path.join(comp_data_dir, str(block_idx) + '-{}.data'.format(col_idx))
                    ndb_utils.save_byte_to_disk(file_name, data_bytes)
                    data_size += sys.getsizeof(data_bytes)
                    list_delta_enabled.append(False)
                else:
                    if list_type[col_idx] != np.int32:
                        data_bytes = col_val.tobytes()
                        file_name = os.path.join(comp_data_dir, str(block_idx) + '-{}.data'.format(col_idx))
                        ndb_utils.save_byte_to_disk(file_name, data_bytes)
                        data_size += sys.getsizeof(data_bytes)
                        list_delta_enabled.append(False)
                    else:
                        # delta
                        col_diff = np.diff(col_val)
                        col_diff_16 = col_diff.astype(diff_dtype)
                        if (col_diff != col_diff_16).any():
                            print('[DELTA WARN] Fine grain Error')
                            diff_dtype = np.int16 
                            col_diff = np.diff(col_val)
                            col_diff_16 = col_diff.astype(diff_dtype)
                            if (col_diff != col_diff_16).any():
                                diff_dtype = np.int32 
                                col_diff = np.diff(col_val)
                                col_diff_16 = col_diff.astype(diff_dtype)

                        file_name = os.path.join(comp_data_dir, str(block_idx) + '-{}.data'.format(col_idx))
                        ndb_utils.save_byte_to_disk(file_name, col_diff_16.tobytes())
                        
                        init_value = np.array(col_val[0], dtype=np.int32)
                        file_name = os.path.join(comp_data_dir, str(block_idx) + '-{}-init.data'.format(col_idx))
                        ndb_utils.save_byte_to_disk(file_name, init_value.tobytes())
                        
                        data_size += col_diff_16.nbytes + init_value.nbytes
                        list_delta_enabled.append(True)

        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = data_size/1024/1024       
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_size/1024/1024))
        np.save(os.path.join(comp_data_dir, 'list_delta_enabled'), list_delta_enabled)
        exp_data_dict['num_record_per_part'] = num_record_per_part 
        exp_data_dict['data_ori_size'] = data_ori_size
        exp_data_dict['data_comp_size'] = data_comp_size
        exp_data_dict['x_start'] = x_start 
        exp_data_dict['x_end'] = x_end 
        exp_data_dict['list_type'] = list_type
        exp_data_dict['diff_dtype'] = diff_dtype
        ndb_utils.save_obj_to_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'), exp_data_dict)
        list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
    else:
        list_delta_enabled = np.load(os.path.join(comp_data_dir, 'list_delta_enabled.npy'))  
        exp_data_dict = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'))
        num_record_per_part = exp_data_dict['num_record_per_part']
        data_ori_size = exp_data_dict['data_ori_size']
        data_comp_size = exp_data_dict['data_comp_size']
        x_start = exp_data_dict['x_start']
        x_end = exp_data_dict['x_end']
        diff_dtype = exp_data_dict['diff_dtype']
        list_type = exp_data_dict['list_type']
        list_sample_index = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(root_path, task_name, 'sample_index_{}.data'.format(sample_size)))

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
                    block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                    block_data = np.frombuffer(block_bytes, dtype=list_type[0])   
                    curr_decomp_block = np.ndarray((len(block_data),), dtype=data_ori.dtype)
                    decomp_memory += sys.getsizeof(block_bytes)
                    curr_decomp_block[curr_decomp_block.dtype.names[0]] = block_data
                    
                    for i in range(1, len(list_delta_enabled)):
                        col_name = data_ori.dtype.names[i]

                        if list_delta_enabled[i] == False:
                            file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(i))
                            block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                            decomp_memory += sys.getsizeof(block_bytes)

                            if list_type[i] == np.int32 or list_type[i] == np.float64:
                                block_data = np.frombuffer(block_bytes, dtype=list_type[i])
                            else:
                                block_data = np.frombuffer(block_bytes, dtype=list_type[i])[col_name]

                            curr_decomp_block[col_name] = block_data
                        else:
                            # delta
                            file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}.data'.format(i))
                            delta_data = np.frombuffer(ndb_utils.read_bytes_from_disk(file_name), dtype=diff_dtype)                      
                            file_name = os.path.join(comp_data_dir, str(part_idx) + '-{}-init.data'.format(i))
                            init_value = np.frombuffer(ndb_utils.read_bytes_from_disk(file_name), dtype=np.int32)                           
                            col_value = np.zeros(len(delta_data)+1, dtype=np.int32)
                            curr_value = init_value[0]

                            for i in range(0, len(delta_data)):
                                col_value[i]  = curr_value
                                curr_value += delta_data[i]    

                            curr_decomp_block[col_name] = col_value
                            decomp_memory += delta_data.nbytes
                            decomp_memory += init_value.nbytes
                            decomp_memory += col_value.nbytes
                            
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
    arr_result = result.copy()
    arr_latency = np.array((data_ori_size, data_comp_size, sample_size, peak_memory/1024/1024, t_sort / num_loop, 0, 0,
    t_locate_part / num_loop, t_decomp / num_loop,  0 / num_loop, # build_index time
    t_lookup / num_loop, 0, 0, t_total / num_loop, num_decomp, count_nonexist)).T

    return_latency = arr_latency.reshape((1,-1))


    return data_ori_size, data_comp_size, [arr_result], return_latency