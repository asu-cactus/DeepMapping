import pandas as pd 
import numpy as np
import sys
import math
import os
from DeepMapping import ndb_utils
# from DeepMapping.ndb_utils import Timer, recreate_temp_dir, save_byte_to_disk, read_bytes_from_disk
from more_itertools import run_length
from tqdm.auto import tqdm




# domain guided encoding

def min_encode_bit(arr):
    # obtain the minimun bits required for encoding
    # check paper Efﬁcient Query Processing with Optimistically Compressed Hash Tables & Strings in the USSR Figure 2
    # return minimun value of the array, maximum value of the array, minimun bits required to encode the arr
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    min_bit = int(np.ceil(np.log2(arr_max - arr_min + 1)))
    return arr_min, arr_max, min_bit

def dge_required_bits(data):
    num_cols = data.shape[1]
    
    required_bits = []
    dge_col_min_vals = []
    for col_idx in range(num_cols):
        arr_min, arr_max, min_bit = min_encode_bit(data[:, col_idx])
        print('Col {} requires {} bits'.format(col_idx, min_bit))
        required_bits.append(min_bit)
        dge_col_min_vals.append(np.min(data[:, col_idx]))

    print('DGE required bits', int(np.ceil(np.sum(required_bits) / 32) * 32))
    return np.array(required_bits, dtype=np.int32), np.array(dge_col_min_vals, dtype=np.int32)

def dge_compression(data, required_bits, dge_col_min_vals):
    num_cols = data.shape[1]
    
    total_required_bytes = int(np.ceil(np.sum(required_bits) / 32) * 32)
    # compression
    if total_required_bytes == 0:
        return np.array(data[0], dtype=np.int32)
    if total_required_bytes == 32:
        dge_comp_data = np.zeros(len(data), dtype=np.uint32)
    elif total_required_bytes == 64:
        dge_comp_data = np.zeros(len(data), dtype=np.uint64)
    for idx, row in tqdm(enumerate(data), total=len(data)):
        current_pack_value = ''
        for col_idx in range(num_cols):
            formatter = '{' + '0:0{}b'.format(required_bits[col_idx]) + '}'
            encoded = row[col_idx] - dge_col_min_vals[col_idx]
            format_value = formatter.format(encoded)
            current_pack_value += format_value
        dge_comp_data[idx] = int(current_pack_value, 2)
    return dge_comp_data

def dge_decompression(dge_comp_data, required_bits, dge_col_min_vals):
    decompressed_value = np.zeros((len(dge_comp_data), len(required_bits)), dtype=np.int32)
    total_required_bits = np.sum(required_bits)
    formatter = '{' + '0:0{}b'.format(total_required_bits) + '}'
    col_bits_ranges = []
    cur_shift_start_bits = 0
    for col_idx in range(len(required_bits)):
        cur_shift_end_bits = cur_shift_start_bits + required_bits[col_idx]
        col_bits_ranges.append([cur_shift_start_bits,cur_shift_end_bits])
        cur_shift_start_bits = cur_shift_end_bits
        
    for idx, val in enumerate(dge_comp_data):
        cur_shift_end_bits = -0
        val_bin = formatter.format(val)
        for col_idx in range(len(required_bits)):
            cur_shift_start_bits, cur_shift_end_bits = col_bits_ranges[col_idx]
            col_val = val_bin[cur_shift_start_bits:cur_shift_end_bits]
            decompressed_value[idx, col_idx] = int(col_val, 2) + dge_col_min_vals[col_idx]
    return decompressed_value


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
    data_ori_size = 0
    data_comp_size = 0
    arr_latency = None 
    arr_result = None

    non_int_cols = []
    non_int_cols_dtypes = []
    int_cols = []

    for col in df.columns:

        if not pd.api.types.is_integer_dtype(df[col]):
            non_int_cols.append(col)
            non_int_cols_dtypes.append(data_ori[col].dtype)
        elif pd.api.types.is_integer_dtype(df[col]):
            int_cols.append(col)

    non_int_data = df[non_int_cols].to_records(index=False) if len(non_int_cols) != 0 else np.array([])
    int_data = df[int_cols].values
    required_bits, dge_col_min_vals = dge_required_bits(int_data)
    dge_comp_data = dge_compression(int_data, required_bits, dge_col_min_vals)

    dge_dtype = np.uint32
    if int(np.ceil(np.sum(required_bits) / 32) * 32) > 32:
        dge_dtype = np.uint64

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
    folder_name = 'dgpe'
    comp_data_dir = os.path.join(root_path, task_name, folder_name, str(block_size))
    print('[Generate File Path]: {}'.format(comp_data_dir))
    # generate file
    if generate_file:
        ndb_utils.recreate_temp_dir(comp_data_dir)
        data_size = 0
        data_partition_idx = (x - x_start) // num_record_per_part
        for block_idx in tqdm(range(num_partition)):
            data_idx = data_partition_idx == block_idx
            data_dge_part = dge_comp_data[data_idx]
            if len(data_dge_part) == 0:
                continue

            if len(non_int_data) != 0:
                data_non_int_part = non_int_data[data_idx]
                data_non_int_part_bytes = data_non_int_part.tobytes()
                file_name1 = os.path.join(comp_data_dir, str(block_idx) + '-nonint.data')
                ndb_utils.save_byte_to_disk(file_name1, data_non_int_part_bytes)
                data_size += sys.getsizeof(data_non_int_part_bytes)
            
            data_dge_part_bytes = data_dge_part.tobytes()      
            data_size += sys.getsizeof(data_dge_part_bytes)       
            file_name2 = os.path.join(comp_data_dir, str(block_idx) + '-dge.data')   
            ndb_utils.save_byte_to_disk(file_name2, data_dge_part_bytes)
        
        data_ori_size = data_ori.nbytes/1024/1024
        data_comp_size = data_size/1024/1024
        print('Ori Size: {}, Curr Size: {}'.format(data_ori.nbytes/1024/1024, data_size/1024/1024))
    
    list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)

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
            current_memory = 0

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
                    file_name2 = os.path.join(comp_data_dir, str(part_idx) + '-dge.data')
                    block_dge_bytes = ndb_utils.read_bytes_from_disk(file_name2)
                    decomp_memory += sys.getsizeof(block_dge_bytes)
                    data_dge_part = np.frombuffer(block_dge_bytes, dtype=dge_dtype)
                    data_int_part = dge_decompression(data_dge_part, required_bits, dge_col_min_vals)
                    decomp_memory += data_int_part.nbytes
                    curr_decomp_block = np.recarray((len(data_dge_part),), dtype=data_ori.dtype)
                    
                    if len(non_int_data) != 0:
                        file_name1 = os.path.join(comp_data_dir, str(part_idx) + '-nonint.data')
                        block_nonint_bytes = ndb_utils.read_bytes_from_disk(file_name1)
                        decomp_memory += sys.getsizeof(block_nonint_bytes)
                        data_non_int_part = np.rec.array(block_nonint_bytes, dtype=non_int_data.dtype)
                        current_memory += sys.getsizeof(data_non_int_part)
                        for i in range(len(non_int_cols)):
                            curr_decomp_block[non_int_cols[i]] = data_non_int_part[non_int_cols[i]]
                        
                        peak_memory += data_non_int_part.nbytes
                                    
                    for i in range(len(int_cols)):
                        curr_decomp_block[int_cols[i]] = data_int_part[:, i]
                    
                    for i in range(len(non_int_cols)):
                        curr_decomp_block[non_int_cols[i]] = data_non_int_part[non_int_cols[i]]
                    
                    decomp_block[part_idx] = curr_decomp_block
                    num_decomp += 1
                    cache_block_memory += curr_decomp_block.nbytes
                else:
                    curr_decomp_block = decomp_block[part_idx]
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
                result_idx += 1
                
                if cache_block_memory + decomp_memory > peak_memory:
                    peak_memory = cache_block_memory + decomp_memory
            t_total += timer_total.toc()
    arr_result = result.copy()
    arr_latency = np.array((data_ori_size, data_comp_size, sample_size, peak_memory/1024/1024, t_sort / num_loop, 
    0, 0, t_locate_part / num_loop, t_decomp / num_loop, 
    t_lookup / num_loop, 0, 0, t_total / num_loop, num_decomp, count_nonexist)).T

    return_latency = arr_latency.reshape((1,-1))

    return data_ori_size, data_comp_size, [arr_result], return_latency