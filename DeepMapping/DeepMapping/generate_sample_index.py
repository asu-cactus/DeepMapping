import argparse
import os
import itertools
from DeepMapping import ndb_utils

# Generate the sample index for TPC-H, SF=1 experiments for re-use in non_generate_file mode

def generate_index(dataset_name, gen_point=False, gen_range=False, change_ratio=False): 
    root_dir = 'temp'
    mode = 'edge'
    uncompress_metadata_path = 'uncompress/131072/extra_meta.data'

    if mode == 'tuning':
        root_dir = 'temp_tune'
        uncompress_metadata_path = 'uncompress/131072/extra_meta.data'

    if dataset_name == 'tpc':
        list_dataset = ['tpch-s1/customer', 'tpch-s1/lineitem', 'tpch-s1/orders', 'tpch-s1/part', 'tpch-s1/supplier',
        'tpch-s10/customer', 'tpch-s10/lineitem', 'tpch-s10/orders', 'tpch-s10/part', 'tpch-s10/supplier']
        list_dataset += ['tpcds-s10/catalog_page', 'tpcds-s10/catalog_returns', 'tpcds-s10/catalog_sales', 'tpcds-s10/customer_address',
        'tpcds-s10/customer_demographics', 'tpcds-s10/customer', 'tpcds-s10/item', 'tpcds-s10/store_returns',
        'tpcds-s10/web_returns']
        uncompress_metadata_path = 'uncompress/131072/extra_meta.data'
    elif dataset_name == 'synthetic':
        list_dataset = [
                    'data_manipulation/single_column_low_corr_100m', 
                    'data_manipulation/single_column_high_corr_100m', 
                    'data_manipulation/multi_column_low_corr_100m', 
                    'data_manipulation/multi_column_high_corr_100m',
                    'data_manipulation/single_column_low_corr_1000m', 
                    'data_manipulation/single_column_high_corr_1000m', 
                    'data_manipulation/multi_column_low_corr_1000m', 
                    'data_manipulation/multi_column_high_corr_1000m',
                    'data_manipulation/single_column_low_corr_10000m', 
                    'data_manipulation/single_column_high_corr_10000m', 
                    'data_manipulation/multi_column_low_corr_10000m', 
                    'data_manipulation/multi_column_high_corr_10000m',
                    ]
        if change_ratio:
            uncompress_metadata_path = 'uncompress/131072/Default/0/extra_meta.data'
        else:
            uncompress_metadata_path = 'uncompress/131072/Default/extra_meta.data'

    if gen_point == True:
        for dataset in list_dataset:
            try:
                path_to_meta = os.path.join(root_dir, dataset, uncompress_metadata_path)
                extra_meta = ndb_utils.load_obj_from_disk_with_pickle(path_to_meta)
                print('[INFO] Generating sample index for', dataset)
                x_start = extra_meta['x_start']
                x_end = extra_meta['x_end']
                num_query = 5
                for sample_size in [1, 100, 1000, 10000, 100000]:
                    list_sample_index = ndb_utils.generate_query(x_start, x_end, num_query=num_query, sample_size=sample_size)
                    
                    ndb_utils.save_obj_to_disk_with_pickle(os.path.join(root_dir, dataset, 'sample_index_{}.data'.format(sample_size)),
                                                        list_sample_index)
            except Exception as e:
                print(e)

parser = argparse.ArgumentParser(description='generate range query or query index')

# Add command-line arguments
parser.add_argument('--dataset', type=str, help='tpc or synthetic', default='tpch')
parser.add_argument('--gen_point', action='store_true', help='Whether to generate point lookup', default=False)
parser.add_argument('--change_ratio', action='store_true', help='Whether is the case for change ratio', default=False)

args = parser.parse_args()
dataset_name = args.dataset
gen_point = args.gen_point
change_ratio = args.change_ratio

print("[INFO]: generate dataset: {}, gen_point: {}, gen_range: {}, change_ratio: {}".format(dataset_name, gen_point, gen_range, change_ratio))
generate_index(dataset_name, gen_point, change_ratio)