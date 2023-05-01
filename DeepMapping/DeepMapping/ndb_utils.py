import numpy as np
import time
import os
import shutil
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



def generate_query(x_start, x_end, num_query=5, sample_size=1000):
    list_sample_index = []
    for query_idx in tqdm(range(num_query)):
        np.random.seed(query_idx)
        sample_index = np.random.choice(np.arange(x_start, x_end+1, dtype=np.int32), sample_size, replace=False).astype(np.int32)
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