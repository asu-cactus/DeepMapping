import numpy as np
import pandas as pd
import time
import os
import shutil
import pandas as pd
from tqdm.auto import tqdm 

def df_preprocess(df, benchmark = None):
    if 'PARTKEY' in df.columns and 'SUPPKEY' in df.columns:
        df.reset_index(inplace=True)
        df.drop(['PARTKEY', 'SUPPKEY', 'QUANTITY'], axis=1, inplace=True)
    elif ('ORDERKEY' in df.columns and benchmark == 'dgpe') or ('ORDERKEY' in df.columns and "deepmapping" in benchmark):
        df.drop(['SHIP-PRIORITY'], axis=1, inplace=True)
    elif 'CR_REFUNDED_CUSTOMER_SK' in df.columns and 'CR_ORDER_NUMBER' in df.columns:
        df.drop(['CR_REFUNDED_CUSTOMER_SK', 'CR_ORDER_NUMBER'], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif 'CS_SOLD_ITEM_SK' in df.columns and 'CS_ORDER_NUMBER' in df.columns:
        df.drop(['CS_SOLD_ITEM_SK', 'CS_ORDER_NUMBER'], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif 'SR_ITEM_SK' in df.columns and 'SR_TICKET_NUMBER' in df.columns:
        df.drop(['SR_ITEM_SK', 'SR_TICKET_NUMBER'], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif 'WR_ITEM_SK' in df.columns and 'WR_ORDER_NUMBER' in df.columns:
        df.drop(['WR_ITEM_SK', 'WR_ORDER_NUMBER'], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif 'WS_ITEM_SK' in df.columns and 'WS_ORDER_NUMBER' in df.columns:
        df.drop(['WS_ITEM_SK', 'WS_ORDER_NUMBER'], axis=1, inplace=True)
        df.reset_index(inplace=True)
    # elif 'C_CUSTOMER_SK' in df.columns:
    #     df.drop(['C_LOGIN'], axis=1, inplace=True)
    elif 'CP_DEPARTMENT' in df.columns and 'deepmapping' in benchmark:
        df.drop(['CP_DEPARTMENT'], axis=1, inplace=True)
    elif 'CA_ADDRESS_COUNTRY' in df.columns and ('deepmapping' in benchmark or 'dgpe' in benchmark):
        df.drop(['CA_ADDRESS_COUNTRY'], axis=1, inplace=True)
    elif 'I_CONTAINER' in df.columns and 'deepmapping' in benchmark:
        df.drop(['I_CONTAINER'], axis=1, inplace=True)
    
    for col in df.columns:
        if df[col].dtypes == np.int64:
            df[col] = df[col].astype(np.int32)
            
    key = df.columns[0]
    df.sort_values(by=key, inplace=True)
    data_ori = df.to_records(index=False)
    return df, data_ori

def data_manipulation(df, ops='None'):
    # default use 90% data
    if ops == 'None':
        pass
    elif ops == 'Default':
        percent_data = 0.9
        np.random.seed(0)
        sampled_index = np.zeros(len(df), dtype=bool)
        sampled_index[np.random.choice(len(df), int(len(df)*percent_data), replace=False)] = True
        # old_df = df.copy()
        # new_df = df.iloc[~sampled_index, :].copy()
        df = df.iloc[sampled_index, :].copy()
        # data_ori = df.to_records(index=False)
    elif ops == 'Insert':
        # rest are inserted
        pass
    elif ops == 'Update': 
        percent_data = 0.9
        np.random.seed(0)
        sampled_index = np.zeros(len(df), dtype=bool)
        sampled_index[np.random.choice(len(df), int(len(df)*percent_data), replace=False)] = True
        rest_df = df.iloc[~sampled_index, :].copy()
        df = df.iloc[sampled_index, :].copy()
        update_index = np.zeros(len(df), dtype=bool)
        update_index[np.random.choice(len(df), len(rest_df), replace=False)] = True
        df.iloc[update_index, :] = rest_df
    elif ops == 'Delete':
        percent_data = 0.9
        np.random.seed(0)
        sampled_index = np.zeros(len(df), dtype=bool)
        sampled_index[np.random.choice(len(df), int(len(df)*percent_data), replace=False)] = True
        rest_df = df.iloc[~sampled_index, :].copy()
        df = df.iloc[sampled_index, :].copy()
        delete_index = np.zeros(len(df), dtype=bool)
        delete_index[np.random.choice(len(df), len(rest_df), replace=False)] = True
        df = df[~delete_index]
    key = df.columns[0]
    df.sort_values(by=key, inplace=True)
    data_ori = df.to_records(index=False)
    return df, data_ori

def process_df_for_synthetic_data(df):
    for col in df.columns:
        if df[col].dtypes == np.int64:
            df[col] = df[col].astype(np.int32)
    return df

def generate_synthetic_data(df_tpch_order, df_tpch_lineitem, df_tpcds_customer_demographics, size=100):
    """This function describes how we generate the synthetic data for our data manipulation experiments
    that cover the four types of data: single column low(high) correlation, multiple columns low(high) 
    correlation. We use the three tables: TPC-H order table (for low correlation), lineitem table and 
    TPC-DS customer demographics table (for high correlation). 
    To generate the low correlation tables, you can generate the order and lineitem tables throughs 
    TPC-H dbgen with specified o -s scale_factor, like 100, 1000 to generate these two tables and send 
    to the script. The script will automatically scale it to the target size (in MB). 
    To generate the high correlation tables, you are required to generate the customer demographics table
    through TPC-DS dsdgen. Since the customer demographics table does not scale with the given scale factor,
    we used the following script to automatically scale it up to the target size by following its pattern.

    Args:
        size (int):
            target size in MB.
    """
    size = size*1024*1024
    # generate single column low correlation
    # single column low correlation is generated based on the TPC-H order table
    df = df_tpch_order.copy()
    df.reset_index(inplace=True)
    df = df[['index', 'ORDERSTATUS']]
    df = process_df_for_synthetic_data(df)
    data_in_recarray = df.to_records(index=False)
    df_single_column_low_correlation = df[:int(np.floor(size/data_in_recarray[0].nbytes))]
    
    # generate multi column low correlation
    # multi column low correlation is generated based on the TPC-H lineitem table
    df = df_tpch_lineitem.copy()
    df.reset_index(inplace=True)
    df = df[['index', 'LINENUMBER', 'DISCOUNT', 'TAX', 'RETURNFLAG', 'LINESTATUS', 'SHIPINSTRUCT', 'SHIPMODE']]
    df = process_df_for_synthetic_data(df)
    data_in_recarray = df.to_records(index=False)
    df_multi_column_low_correlation = df[:int(np.floor(size/data_in_recarray[0].nbytes))]

    # generate single column high correlation
    # single column high correlation is generated based on the TPC-DS customer demographics table
    df = df_tpcds_customer_demographics[['CD_DEMO_SK', 'CD_EDUCATION_STATUS']].copy()
    df = process_df_for_synthetic_data(df)
    df_new = df.copy()
    data_in_recarray = df.to_records(index=False)
    data_size = data_in_recarray.nbytes
    repeat_time = int(np.floor(size/data_size))
    last_id = np.max(df.iloc[:, 0])
    for append_copy_idx in range(1,repeat_time+1):
        append_copy = df.copy()
        cd_demo_sk_copy= np.arange(last_id + (append_copy_idx-1)*len(append_copy), last_id + append_copy_idx*len(append_copy))
        append_copy.loc[:, 'CD_DEMO_SK'] = cd_demo_sk_copy
        df_new = pd.concat((df_new, append_copy))
    df_single_column_high_correlation = df_new[:int(np.floor(size//data_in_recarray[0].nbytes))]
    
    # generate multiple columns high correlation
    # multiple columns high correlation is generated based on the TPC-DS customer demographics table
    df = df_tpcds_customer_demographics.copy()
    df_new = df.copy()
    data_in_recarray = df.to_records(index=False)
    data_size = data_in_recarray.nbytes
    repeat_time = int(np.floor(size/data_size))
    last_id = np.max(df.iloc[:, 0])
    for append_copy_idx in range(1,repeat_time+1):
        append_copy = df[df['cd_dep_college_count'] == 6].copy()
        cd_demo_sk_copy= np.arange(last_id + (append_copy_idx-1)*len(append_copy), last_id + append_copy_idx*len(append_copy))
        append_copy.loc[:, 'cd_demo_sk'] = cd_demo_sk_copy
        append_copy.loc[:, 'cd_dep_college_count'] += append_copy_idx
        df_new = pd.concat((df_new, append_copy))
    df_multi_column_high_correlation = df_new[:int(np.floor(size//data_in_recarray[0].nbytes))]

    return df_single_column_low_correlation, df_single_column_high_correlation, df_multi_column_low_correlation, df_multi_column_high_correlation

def create_features(x, max_len=None):
    if max_len is None:
        max_len = len(str(np.max(x)))
    # one-hot encoding for each digit
    x_features = np.zeros((len(x), max_len*10))
    for idx in range(len(x)):
        digit_idx = max_len - 1
        for digit in str(x[idx])[::-1]:
            x_features[idx, digit_idx*10 + int(digit)] = 1
            digit_idx -= 1
    return x_features, max_len

def generate_query(x_start, x_end, num_query=5, sample_size=1000):
    list_sample_index = []
    for query_idx in tqdm(range(num_query)):
        np.random.seed(query_idx)
        try:
            sample_index = np.random.choice(np.arange(x_start, x_end+1, dtype=np.int32), sample_size, replace=False).astype(np.int32)
        except:
            print("[WARN] Sample size to big, sample with replace instead")
            sample_index = np.random.choice(np.arange(x_start, x_end+1, dtype=np.int32), sample_size, replace=True).astype(np.int32)
        list_sample_index.append(sample_index)
    return list_sample_index

def generate_range_query(x_start, x_end, num_query=5, query_range=1000):
    list_query_range = []
    for query_idx in tqdm(range(num_query)):
        np.random.seed(query_idx)
        query_start = np.random.randint(x_start, x_end, dtype=np.int32)
        query_end = query_start + query_range
        list_query_range.append([query_start, query_end])
    return list_query_range

def delete_file_if_exist(path):
    if os.path.exists(path):
        os.remove(path)

def recreate_temp_dir(comp_data_dir):
    if not os.path.exists(comp_data_dir):
        os.makedirs(comp_data_dir)
    else:
        shutil.rmtree(comp_data_dir)
        os.makedirs(comp_data_dir)
def save_byte_to_disk(file_name, f_bytes):
    with open(file_name, "wb") as binary_file:
        binary_file.write(f_bytes)
def read_bytes_from_disk(file_name):
    with open(file_name, "rb") as f:
        bytes_read = f.read()
    return bytes_read


class Timer(object):
    """A convenient class to measure the running time of a program

    """
    def __init__(self):
        self.start = 0
        self.end = 0
    
    def tic(self):
        """Tic the start time

        """
        self.start = time.perf_counter()
    
    def toc(self):
        """Toc the end time and return the running time

        Returns:
            float: running time (ms)
        """
        self.end = time.perf_counter()
        return (self.end - self.start)*1000

def binary_search(x, val, num_record, search_larger_value=False):
    low = 0
    high = num_record - 1
    while (low <= high):
        mid = low + ((high-low) // 2)
        
        if x[mid] == val:
            return mid
        elif val < x[mid]:
            high = mid - 1
        elif val > x[mid]:
            low = mid + 1
    if search_larger_value:
        return low
    else:
        return -1