import os
import itertools
from DeepMapping import ndb_utils

# Generate the sample index for TPC-H, SF=1 experiments for re-use in non_generate_file mode

list_dataset = ['tpch-s1/customer', 'tpch-s1/lineitem', 'tpch-s1/orders', 'tpch-s1/part', 'tpch-s1/supplier']

for dataset in list_dataset:
    path_to_meta = os.path.join('temp', dataset, 'uncompress/extra_meta.data')
    print('[INFO] Generating sample index for', dataset)
    extra_meta = ndb_utils.load_obj_from_disk_with_pickle(path_to_meta)
    x_start = extra_meta['x_start']
    x_end = extra_meta['x_end']
    num_query = 5
    for sample_size in [1000, 10000, 100000]:
        list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
        
        ndb_utils.save_obj_to_disk_with_pickle(os.path.join('temp', dataset, 'sample_index_{}.data'.format(sample_size)),
                                               list_sample_index)
        


# Generate the sample index for data manipulation experiments for re-use in non_generate_file mode

list_dataset = ['data_manipulation/single_column_low_corr_100m', 
                'data_manipulation/single_column_high_corr_100m', 
                'data_manipulation/multi_column_low_corr_100m', 
                'data_manipulation/multi_column_high_corr_100m']


for dataset in list_dataset:
    path_to_meta = os.path.join('temp', dataset, 'uncompress', 'Default', 'extra_meta.data')
    print('[INFO] Generating sample index for', dataset)
    extra_meta = ndb_utils.load_obj_from_disk_with_pickle(path_to_meta)
    x_start = extra_meta['x_start']
    x_end = extra_meta['x_end']
    num_query = 5
    for sample_size in [1000, 10000, 100000]:
        list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
        
        ndb_utils.save_obj_to_disk_with_pickle(os.path.join('temp', dataset, 'sample_index_{}.data'.format(sample_size)),
                                               list_sample_index)