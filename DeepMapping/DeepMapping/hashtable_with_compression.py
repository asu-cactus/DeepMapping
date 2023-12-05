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
import multiprocessing
from DeepMapping import ndb_utils
from tqdm.auto import tqdm
import concurrent.futures
import multiprocessing.shared_memory as shm

ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.bool_, 
                                            ndim=1,
                                            flags="C")
ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, 
                                    ndim=1,
                                    flags="C")

shared_utils = ctypes.CDLL(os.path.abspath("shared_utils.so")) # Or full path to file 
shared_utils.aux_look_up_bin.argtypes = [ND_POINTER_2, ctypes.c_int, ctypes.c_long]
shared_utils.aux_look_up_bin.restype = ctypes.c_long

def write_partition_to_disk(arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, num_record, block_start_idx, block_end_idx, key, comp_data_dir, zstd_compress_level):
    # Open the shared memory array
    arr_shm = shm.SharedMemory(name=arr_shm_name)
    # Access the shared array as a NumPy array
    data_ori = np.ndarray((num_record,), dtype=arr_shm_dtypes, buffer=arr_shm.buf)

    arr_shm1 = shm.SharedMemory(name=arr_shm_name1)
    data_partition_idx = np.ndarray((num_record,), dtype=arr_shm_dtypes1, buffer=arr_shm1.buf)


    total_size = 0
    enable_progress_bar = block_end_idx >= np.max(data_partition_idx)
    if enable_progress_bar:
        progress_bar = tqdm(range(block_start_idx, block_end_idx), desc="Progress")

    for block_idx in range(block_start_idx, block_end_idx):
        data_idx = data_partition_idx == block_idx
        data_part = data_ori[data_idx]
        if len(data_part) == 0:
            continue
        data_part_hash_table = dict()
        for data_idx in range(len(data_part)):
            data_part_hash_table[data_part[key][data_idx]] = data_part[data_idx]

        data_part_hash_table_bytes = pickle.dumps(data_part_hash_table)
        file_name = os.path.join(comp_data_dir, str(block_idx) + '.data')
        ndb_utils.save_byte_to_disk(file_name, zstd.compress(data_part_hash_table_bytes, zstd_compress_level))
        total_size += os.path.getsize(file_name)
        if enable_progress_bar:
            progress_bar.update(1)
    return total_size

def update_partition_to_disk(arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, num_record, block_start_idx, block_end_idx, comp_data_dir, data_op, key, zstd_compress_level):
    # Open the shared memory array
    arr_shm = shm.SharedMemory(name=arr_shm_name)
    # Access the shared array as a NumPy array
    data_ori = np.ndarray((num_record,), dtype=arr_shm_dtypes, buffer=arr_shm.buf)

    arr_shm1 = shm.SharedMemory(name=arr_shm_name1)
    data_partition_idx = np.ndarray((num_record,), dtype=arr_shm_dtypes1, buffer=arr_shm1.buf)

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
            block_bytes = ndb_utils.read_bytes_from_disk(file_name)
            curr_decomp_block = pickle.loads(zstd.uncompress(block_bytes))
        except:
            data_part_hash_table = dict()
            for data_idx in range(len(data_part)):
                data_part_hash_table[data_part[key][data_idx]] = data_part[data_idx]
            with ndb_utils.get_temp_file_path() as temp_path:
                data_part_hash_table_bytes = pickle.dumps(data_part_hash_table)
                ndb_utils.save_byte_to_disk(temp_path, zstd.compress(data_part_hash_table_bytes, zstd_compress_level))
            continue
        curr_hash_table = curr_decomp_block
        if data_op == 'Insert':
            for idx in range(len(data_part)):
                curr_hash_table[data_part[key][idx]] = data_part[idx]
        elif data_op == 'Update':
            for i in range(len(data_part)):
                if data_part[key][i] in curr_hash_table:
                    curr_hash_table[data_part[key][i]] = data_part[i]
        
        with ndb_utils.get_temp_file_path() as temp_path:
            data_part_hash_table_bytes = pickle.dumps(curr_hash_table)
            ndb_utils.save_byte_to_disk(temp_path, zstd.compress(data_part_hash_table_bytes, zstd_compress_level))
        if enable_progress_bar:
            progress_bar.update(1)
    return total_size

def delete_tuple_in_disk(arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, num_record, num_col, block_start_idx, block_end_idx, comp_data_dir, data_op, key, data_ori_dtype, zstd_compress_level):
    # Open the shared memory array
    arr_shm = shm.SharedMemory(name=arr_shm_name)
    # Access the shared array as a NumPy array
    data_idx_to_delete = np.ndarray((num_record,), dtype=arr_shm_dtypes, buffer=arr_shm.buf)

    arr_shm1 = shm.SharedMemory(name=arr_shm_name1)
    data_idx_to_delete_partition_idx = np.ndarray((num_record,), dtype=arr_shm_dtypes1, buffer=arr_shm1.buf)

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
            block_bytes = ndb_utils.read_bytes_from_disk(file_name)
            curr_decomp_block = pickle.loads(zstd.uncompress(block_bytes))
        except:
            continue
        curr_hash_table = curr_decomp_block
        
        for i in range(len(data_part)):
            if data_part[i] in curr_hash_table:
                del curr_hash_table[data_part[i]]

        with ndb_utils.get_temp_file_path() as temp_path:
            data_part_hash_table_bytes = pickle.dumps(curr_hash_table)
            ndb_utils.save_byte_to_disk(temp_path, zstd.compress(data_part_hash_table_bytes, zstd_compress_level))
        if enable_progress_bar:
            progress_bar.update(1)
    return total_size


def measure_latency(df, data_ori, task_name, sample_size, 
                    generate_file=True,root_path='temp', 
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
            search algorithm that applied to search entry in each partition, available strategy: naive, binary, hash
        block_size  : int
            block size for each partition, size in bytes
    """
    zstd_compress_level = 4
    max_generate_file_threads = int(os.environ['MAX_GENERATE_FILE_THREADS'])
    if 'zstd_compress_level' in kwargs:
        zstd_compress_level = kwargs['zstd_compress_level']
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
    if os.environ['MODE'] == 'tuning':
        root_path = 'temp_tune'
    folder_name = 'hashtable_with_compression'
    comp_data_dir = os.path.join(root_path, task_name, folder_name, str(block_size))
    if 'DATA_OPS' in os.environ and 'CHANGE_RATIO' in os.environ:
        comp_data_dir = os.path.join(comp_data_dir, os.environ['DATA_OPS'], os.environ['CHANGE_RATIO'])
        
    print('[Generate File Path]: {}'.format(comp_data_dir))

    dict_contigous_key = dict()
    
    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        data_size = 0
        data_partition_idx = (x - x_start) // num_record_per_part

        num_threads_to_generate_file = max_generate_file_threads if max_generate_file_threads <= ndb_utils.get_sys_num_threads() else ndb_utils.get_sys_num_threads()
        num_partition_per_core = int(np.ceil(num_partition/num_threads_to_generate_file))
        arr_shm = shm.SharedMemory(create=True, size=data_ori.nbytes)
        shared_array = np.ndarray(data_ori.shape, dtype=data_ori.dtype, buffer=arr_shm.buf)
        shared_array[:] = data_ori[:]
        arr_shm_name = arr_shm.name
        arr_shm_dtypes = data_ori.dtype

        arr_shm1 = shm.SharedMemory(create=True, size=data_partition_idx.nbytes)
        shared_array1 = np.ndarray(data_partition_idx.shape, dtype=data_partition_idx.dtype, buffer=arr_shm1.buf)
        shared_array1[:] = data_partition_idx[:]
        arr_shm_name1 = arr_shm1.name
        arr_shm_dtypes1 = data_partition_idx.dtype

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads_to_generate_file) as executor:
            # Launch threads to process the numbers
            futures = [executor.submit(write_partition_to_disk, arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, data_partition_idx.shape[0], i*num_partition_per_core, (i+1)*num_partition_per_core, key, comp_data_dir, zstd_compress_level) for i in range(num_threads_to_generate_file)]
            # Retrieve the results from each thread
            for future in concurrent.futures.as_completed(futures):
                data_size += future.result()

        arr_shm.close()
        arr_shm.unlink()
        arr_shm1.close()
        arr_shm1.unlink()


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
        arr_result = result.copy()
        del result
        gc.collect()
    arr_latency = np.array((data_ori_size, data_comp_size, sample_size, peak_memory/1024/1024, t_sort / num_loop, 
    0, 0, t_locate_part / num_loop, t_decomp / num_loop, t_build_index / num_loop,
    t_lookup / num_loop, 0, 0, t_total / num_loop, num_decomp, count_nonexist)).T

    return_latency = arr_latency.reshape((1,-1))

    return data_ori_size, data_comp_size, [arr_result], return_latency

def measure_latency_data_update(df, data_ori, data_change, data_op, task_name, sample_size, 
                    generate_file=True,root_path='temp', 
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
            search algorithm that applied to search entry in each partition, available strategy: naive, binary, hash
        block_size  : int
            block size for each partition, size in bytes
    """
    zstd_compress_level = 4
    max_generate_file_threads = int(os.environ['MAX_GENERATE_FILE_THREADS'])
    if 'zstd_compress_level' in kwargs:
        zstd_compress_level = kwargs['zstd_compress_level']
    mode = os.environ['MODE']
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
    if os.environ['MODE'] == 'tuning':
        root_path = 'temp_tune'
    folder_name = 'hashtable_with_compression'
    comp_data_dir = os.path.join(root_path, task_name, folder_name, str(block_size))
        
    print('[Generate File Path]: {}'.format(comp_data_dir))
    timer_update = ndb_utils.Timer()
    t_update = 0
    ori_data_change = data_change.copy()
    for _ in tqdm(range(num_loop)):
        data_change = ori_data_change.copy()

        timer_update.tic()



        exp_data_dict = ndb_utils.load_obj_from_disk_with_pickle(os.path.join(comp_data_dir, 'extra_meta.data'))
        num_record_per_part = exp_data_dict['num_record_per_part']
        data_ori_size = exp_data_dict['data_ori_size']
        data_comp_size = exp_data_dict['data_comp_size']
        x_start = exp_data_dict['x_start']
        x_end = exp_data_dict['x_end']

        if data_op == 'Insert' or data_op == 'Update':
            x = data_change[key]
            x_end = x_end if x_end >= np.max(x) else np.max(x)
            x_start = np.min(x)
            x_range = x_end - x_start
            num_partition = int(math.ceil(x_range / num_record_per_part))
            data_partition_idx = (x - x_start) // num_record_per_part
            data_partition_idx_start = int(x_start // num_record_per_part)
            data_change = data_change.to_records(index=False)
            data_change = ndb_utils.recarr_to_ndarray(data_change)

            num_threads_to_generate_file = max_generate_file_threads if max_generate_file_threads <= ndb_utils.get_sys_num_threads() else ndb_utils.get_sys_num_threads()
            num_partition_per_core = int(np.ceil(num_partition/num_threads_to_generate_file))

            arr_shm = shm.SharedMemory(create=True, size=data_change.nbytes)
            shared_array = np.ndarray(data_change.shape, dtype=data_change.dtype, buffer=arr_shm.buf)
            shared_array[:] = data_change[:]
            arr_shm_name = arr_shm.name
            arr_shm_dtypes = data_change.dtype

            arr_shm1 = shm.SharedMemory(create=True, size=data_partition_idx.nbytes)
            shared_array1 = np.ndarray(data_partition_idx.shape, dtype=data_partition_idx.dtype, buffer=arr_shm1.buf)
            shared_array1[:] = data_partition_idx[:]
            arr_shm_name1 = arr_shm1.name
            arr_shm_dtypes1 = data_partition_idx.dtype


            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads_to_generate_file) as executor:
                # Launch threads to process the numbers
                futures = [executor.submit(update_partition_to_disk, arr_shm_name, arr_shm_dtypes, arr_shm_name1, arr_shm_dtypes1, data_partition_idx.shape[0], data_partition_idx_start+i*num_partition_per_core, data_partition_idx_start+(i+1)*num_partition_per_core, comp_data_dir, data_op, key, zstd_compress_level) for i in range(num_threads_to_generate_file)]
                # Retrieve the results from each thread
                for future in concurrent.futures.as_completed(futures):
                    future.result()

            arr_shm.close()
            arr_shm.unlink()
            arr_shm1.close()
            arr_shm1.unlink()

        elif data_op == 'Delete':
            data_change = data_change.values
            x_start = np.min(data_change)
            x_range = x_end - x_start
            num_partition = int(math.ceil(x_range / num_record_per_part))
            data_idx_to_delete = data_change
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
                                        arr_shm_name3, arr_shm_dtypes3, data_idx_to_delete.shape[0], data_idx_to_delete.shape[1], data_partition_idx_start+i*num_partition_per_core, data_partition_idx_start+(i+1)*num_partition_per_core, comp_data_dir, data_op, key, data_ori.dtype, zstd_compress_level) for i in range(num_threads_to_generate_file)]
                # Retrieve the results from each thread
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            arr_shm2.close()
            arr_shm2.unlink()
            arr_shm3.close()
            arr_shm3.unlink()
        else:
            raise ValueError('Non-supported data-ops')

        t_update += timer_update.toc()
    
    t_update /= num_loop
    return t_update
