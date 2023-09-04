import os
# -1 disable gpu, 0 use gpu 0, 1 use gpu 1
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd 
import itertools
from tqdm.auto import tqdm
from DeepMapping.ndb_utils import df_preprocess
from DeepMapping.benchmark_utils import benchmark_handler

list_dataset = ['tpch-s1/customer', 'tpch-s1/lineitem', 'tpch-s1/orders', 'tpch-s1/part', 'tpch-s1/supplier']
list_benchmark = ['uncompress', 'zstd', 'deepmapping', 'hashtable', 'hashtable_with_compression']
list_sample_size = [1000, 100000]
list_run_config = list(itertools.product(list_dataset, list_benchmark, list_sample_size))
print('[Config]: \n\t Dataset: {} \n\t Benchmark: {} \n\t Sample Size: {}'.format(list_dataset, list_benchmark, list_sample_size))

num_loop = 100
num_query = 5
search_algo = 'binary'
file_name = 'benchmark_data_query.csv'
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
os.environ['MODE'] = 'full'

for run_config in tqdm(list_run_config):
    print('[STATUS] Current config: {}'.format(run_config))
    task_name, benchmark, sample_size = run_config
    if generate_file: 
        df = pd.read_csv('dataset/{}.csv'.format(task_name))
    else:
        df = pd.read_csv('dataset/{}.csv'.format(task_name), nrows=2)
    df, data_ori = df_preprocess(df, benchmark)
    function_call = benchmark_handler(benchmark)

    try:
        data_ori_size, data_comp_size, result, latency = function_call(df=df, data_ori=data_ori, 
                                                                task_name=task_name,  sample_size=sample_size,
                                                                generate_file=generate_file, num_loop=num_loop,
                                                                num_query=num_query, search_algo=search_algo) 
        result_df = pd.DataFrame(latency)
        result_df['config'] = str(run_config)
        result_df['search'] = search_algo
        result_df['size'] = str(data_comp_size)
        
    except Exception as e:
        print('[ERROR] Error in config: {}, Message: {}'.format(run_config, e))
        result_df = pd.DataFrame(latency)
        result_df['config'] = str(run_config)
        result_df['error'] = e

    result_df.to_csv(file_name, mode='a', index=False, header=False)