import gc
import lzo
import math
import numpy as np
import os
import pandas as pd 
import sys
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
    arr_latency = None 
    arr_result = None
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
    if os.environ['MODE'] == 'tuning':
        root_path = 'temp_tune'
    folder_name = 'lzo'
    comp_data_dir = os.path.join(root_path, task_name, folder_name, str(block_size))
    print('[Generate File Path]: {}'.format(comp_data_dir))

    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        data_size = 0
        data_partition_idx = (x - x_start) // num_record_per_part
        for block_idx in tqdm(range(num_partition)):
            data_idx = data_partition_idx == block_idx
            data_part = data_ori[data_idx]

            if len(data_part) == 0:
                continue
            data_bytes = data_part.tobytes()
            data_lzo_comp = lzo.compress(data_bytes, 1)
            data_size += sys.getsizeof(data_lzo_comp)
            file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')
            ndb_utils.save_byte_to_disk(file_name, data_lzo_comp)

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
    timer_locate_part = ndb_utils.Timer()
    timer_build_index = ndb_utils.Timer()
    t_decomp = 0
    t_lookup = 0
    t_sort = 0
    t_total = 0
    t_locate_part = 0
    t_build_index = 0
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
            sample_index_partition = (sample_index_sorted - x_start) // num_record_per_part
            sample_index_partition = sample_index_partition.astype(np.int32)
            t_sort += timer_sort.toc()
            result = np.ndarray((sample_size,), dtype=data_ori.dtype)
            result_idx = 0

            for idx in range(sample_size):
                query_key = sample_index_sorted[idx]
                query_key_index_in_old = sample_index_argsort[idx]        
                timer_locate_part.tic()                 
                part_idx = sample_index_partition[idx]
                # part_idx = int((query_key-x_start) // num_record_per_part)
                t_locate_part += timer_locate_part.toc()
                timer_decomp.tic()

                if part_idx not in decomp_block:
                    if mode == 'edge':
                        available_memory = ndb_utils.get_available_memory()
                        if available_memory < 1024*1024*100:
                            # memory not eneough, free some memory
                            curr_decomp_block = ndb_utils.evict_unused_partition(curr_decomp_block, partition_hit, free_memory=1024*1024*100)

                    partition_hit[part_idx] =1

                    file_name = os.path.join(comp_data_dir, str(part_idx) + '.data')
                    block_bytes = ndb_utils.read_bytes_from_disk(file_name)
                    curr_decomp_block = np.frombuffer(lzo.decompress(block_bytes), dtype=data_ori.dtype)
                    decomp_block[part_idx] = curr_decomp_block
                    num_decomp += 1
                    cache_block_memory += curr_decomp_block.nbytes
                    block_bytes_size = sys.getsizeof(block_bytes)
                    if search_algo == 'hash':
                        t_decomp += timer_decomp.toc()
                        timer_build_index.tic()
                        t_build_index += timer_build_index.toc()
                        timer_decomp.tic()
                else:
                    curr_decomp_block = decomp_block[part_idx]
                    partition_hit[part_idx] +=1
                
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
                if cache_block_memory + block_bytes_size > peak_memory:
                    peak_memory = cache_block_memory + block_bytes_size
            t_total += timer_total.toc()
        arr_result = result.copy()
        del result
        gc.collect()
    arr_latency = np.array((data_ori_size, data_comp_size, sample_size, peak_memory/1024/1024, t_sort / num_loop, 
    0, 0, t_locate_part / num_loop, t_decomp / num_loop, t_build_index / num_loop,
    t_lookup / num_loop, 0, 0, t_total / num_loop, num_decomp, count_nonexist)).T

    return_latency = arr_latency.reshape((1,-1))

    return data_ori_size, data_comp_size, [arr_result], return_latency