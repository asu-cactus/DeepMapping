import os
# -1 disable gpu, 0 use gpu 0, 1 use gpu 1
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd 
import itertools
import numpy as np
from tqdm.auto import tqdm
from DeepMapping import ndb_utils
from DeepMapping.ndb_utils import df_preprocess
from DeepMapping.benchmark_utils import benchmark_handler

list_dataset = ['tpch-s1/customer', 'tpch-s1/lineitem', 'tpch-s1/orders', 'tpch-s1/part', 'tpch-s1/supplier']
list_benchmark = ['uncompress', 'zstd', 'deepmapping', 'hashtable', 'hashtable_with_compression']
list_block_size = [1024*128, 1024*256, 1024*512, 1024*1024, 1024*2048, 1024*4096, 1024*4096*2]
list_zstd_compress_level = range(1, 23)
list_sample_size = [1000, 10000, 100000]
list_run_config = list(itertools.product(list_dataset, list_benchmark, list_sample_size, list_block_size, list_zstd_compress_level))
print('[Config]: \n\t Dataset: {} \n\t Benchmark: {} \n\t Sample Size: {} \n\t Block Size: {}'.format(list_dataset, list_benchmark, list_sample_size, list_block_size))

num_loop = 5
num_query = 1
search_algo = 'binary'
file_name = 'benchmark_data_grid_search.csv'
# The following flag is used to indicate whether you can re-use the existing disk 
# files (stored in temp dir) saved from a fresh run. Usually, you can start a 
# fresh run and then change this flag to False. Also, if you set this flag a False
# please make sure, you also run the generate_sample_index.py under DeepMapping
# folder to pre-generate the query index before your next run.
generate_file = True   
# specificy your deep learning model backend, current support keras h5 model and onnx
# model. There is a utility under DeepMapping to convert a h5 model into onnx format.
os.environ['BACKEND'] = 'onnx'
# Run the benchmark with the specified mode. full mode: assume memory is sufficient to cache
# all the data; edge mode: try to cache all data within the available memory but reserve
# a number of free memory for underlying process, current value: 100MB. Once the memory 
# is insufficient, it will try to evict the least used partition to free the memory.
os.environ['MODE'] = 'tuning'
os.environ['MAX_GENERATE_FILE_THREADS'] = '2'

for run_config in tqdm(list_run_config):
    print('[STATUS] Current config: {}'.format(run_config))
    task_name, benchmark, sample_size = run_config
    task_name, benchmark, sample_size, block_size, zstd_compress_level = run_config
    if os.environ['MODE'] == 'edge':
        if generate_file == True:
            raise ValueError("MODE: edge is only used for benchmark purpose cannot used with generate_file")
        df = pd.read_csv('dataset/{}.csv'.format(task_name), nrows=10)
    elif os.environ['MODE'] == 'full':
        df = pd.read_csv('dataset/{}.csv'.format(task_name))
    elif os.environ['MODE'] == 'tuning':
        df = pd.read_csv('dataset/{}.csv'.format(task_name), nrows=10)
        df, data_ori = df_preprocess(df, benchmark)
        size_per_tuple = data_ori[0].nbytes
        # use 10 MB dataset for benchmark
        nrows_to_load = int(np.floor(1024*1024*10/size_per_tuple))
        df = pd.read_csv('dataset/{}.csv'.format(task_name), nrows=nrows_to_load)
    df, data_ori = df_preprocess(df, benchmark)
    function_call = benchmark_handler(benchmark)

    try:
        data_ori_size, data_comp_size, result, latency = function_call(df=df, data_ori=data_ori, 
                                                                task_name=task_name,  sample_size=sample_size,
                                                                generate_file=generate_file, num_loop=num_loop,
                                                                num_query=num_query, search_algo=search_algo,
                                                                block_size=block_size, zstd_compress_level=zstd_compress_level) 
        if benchmark == 'deepmapping':
            result_df = pd.DataFrame(latency[:, :-2])
        else:
            result_df = pd.DataFrame(latency)
        result_df['config'] = str(run_config)
        result_df['search'] = search_algo
        result_df['block_size_mb'] = str(block_size/1024/1024)
        result_df['zstd_compress_level'] = zstd_compress_level
        result_df['size'] = str(data_comp_size) 
        if benchmark == 'deepmapping':
            result_df['bit_array_size'] = str(latency[:, -2])
            result_df['model_size'] = str(latency[:, -1])
        
    except Exception as e:
        print('[ERROR] Error in config: {}, Message: {}'.format(run_config, e))
        result_df = pd.DataFrame(latency)
        result_df['config'] = str(run_config)
        result_df['error'] = e

    result_df.to_csv(file_name, mode='a', index=False, header=False)