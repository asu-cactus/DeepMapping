import os
# -1 disable gpu, 0 use gpu 0, 1 use gpu 1
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd 
import itertools
from tqdm.auto import tqdm
from DeepMapping.ndb_utils import df_preprocess, data_manipulation
from DeepMapping.benchmark_utils import benchmark_handler

list_dataset = ['data_manipulation/single_column_low_corr_100m', 
                'data_manipulation/single_column_high_corr_100m', 
                'data_manipulation/multi_column_low_corr_100m', 
                'data_manipulation/multi_column_high_corr_100m']
list_benchmark = ['uncompress', 'zstd', 'deepmapping']
list_sample_size = [10000]
list_ops = ['Default', 'Insert', 'Update', 'Delete']
list_run_config = list(itertools.product(list_dataset, list_benchmark, list_sample_size, list_ops))
print('[Config]: \n\t Dataset: {} \n\t Benchmark: {} \n\t Sample Size: {}'.format(list_dataset, list_benchmark, list_sample_size))

memory_optimized = False
latency_optimized = True
num_loop = 100
num_query = 5
search_algo = 'binary'
file_name = 'benchmark_data_manipulation.csv'

for run_config in tqdm(list_run_config):
    print('[STATUS] Current config: {}'.format(run_config))
    task_name, benchmark, sample_size, data_ops = run_config   
    generate_file = True

    df = pd.read_csv('dataset/{}.csv'.format(task_name))
    df, data_ori = df_preprocess(df, benchmark)
    # perform data manipulation to the data
    df, data_ori = data_manipulation(df, data_ops)
    function_call = benchmark_handler(benchmark)
    
    try:
        data_ori_size, data_comp_size, result, latency = function_call(df=df, data_ori=data_ori, 
                                                                task_name=task_name,  sample_size=sample_size,
                                                                generate_file=generate_file, memory_optimized=memory_optimized,
                                                                latency_optimized=latency_optimized, num_loop=num_loop,
                                                                num_query=num_query, search_algo=search_algo) 
        result_df = pd.DataFrame(latency)
        result_df['config'] = str(run_config)
        result_df['data-ops'] = str(data_ops)
        result_df['search'] = search_algo
        result_df['size'] = str(data_comp_size)
           
    except Exception as e:
        print('[ERROR] Error in config: {}, Message: {}'.format(run_config, e))
        result_df = pd.DataFrame(latency)
        result_df['config'] = str(run_config)
        result_df['error'] = e

    result_df.to_csv(file_name, mode='a', index=False, header=False)
