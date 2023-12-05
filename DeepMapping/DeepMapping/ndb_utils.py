import gc
import numpy as np
import os
import pandas as pd
import pickle
import psutil
import shutil
import time
import sys
import multiprocessing
import tempfile
from heapq import nsmallest
from tqdm.auto import tqdm


def recarr_to_ndarray(rec_arr):
    type_list = []
    for i in range(len(rec_arr.dtype)):
        col_name = rec_arr.dtype.names[i]
        col_type = rec_arr.dtype[i]

        if col_type == "O":
            col_type = "S8"
        type_list.append((col_name, col_type))

    ndarray = np.array(rec_arr, dtype=type_list)
    del rec_arr
    return ndarray


def df_preprocess(df, benchmark=None, to_ndarray=True, is_data_manipulation=False):
    if "PARTKEY" in df.columns and "SUPPKEY" in df.columns:
        df.reset_index(inplace=True)
        df.drop(["PARTKEY", "SUPPKEY", "QUANTITY"], axis=1, inplace=True)
    elif ("ORDERKEY" in df.columns and benchmark == "dgpe") or (
        "ORDERKEY" in df.columns and "deepmapping" in benchmark
    ):
        df.drop(["SHIP-PRIORITY"], axis=1, inplace=True)
    elif "CR_REFUNDED_CUSTOMER_SK" in df.columns and "CR_ORDER_NUMBER" in df.columns:
        df.drop(["CR_REFUNDED_CUSTOMER_SK", "CR_ORDER_NUMBER"], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif "CS_SOLD_ITEM_SK" in df.columns and "CS_ORDER_NUMBER" in df.columns:
        df.drop(["CS_SOLD_ITEM_SK", "CS_ORDER_NUMBER"], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif "SR_ITEM_SK" in df.columns and "SR_TICKET_NUMBER" in df.columns:
        df.drop(["SR_ITEM_SK", "SR_TICKET_NUMBER"], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif "WR_ITEM_SK" in df.columns and "WR_ORDER_NUMBER" in df.columns:
        df.drop(["WR_ITEM_SK", "WR_ORDER_NUMBER"], axis=1, inplace=True)
        df.reset_index(inplace=True)
    elif "WS_ITEM_SK" in df.columns and "WS_ORDER_NUMBER" in df.columns:
        df.drop(["WS_ITEM_SK", "WS_ORDER_NUMBER"], axis=1, inplace=True)
        df.reset_index(inplace=True)
    # elif 'C_CUSTOMER_SK' in df.columns:
    #     df.drop(['C_LOGIN'], axis=1, inplace=True)
    elif "CP_DEPARTMENT" in df.columns and "deepmapping" in benchmark:
        df.drop(["CP_DEPARTMENT"], axis=1, inplace=True)
    elif "CA_ADDRESS_COUNTRY" in df.columns and (
        "deepmapping" in benchmark or "dgpe" in benchmark
    ):
        df.drop(["CA_ADDRESS_COUNTRY"], axis=1, inplace=True)
    elif "I_CONTAINER" in df.columns and "deepmapping" in benchmark:
        df.drop(["I_CONTAINER"], axis=1, inplace=True)

    df = change_df_int64_to_int32(df)

    key = df.columns[0]
    if is_data_manipulation:
        # for data manipulation case, no need to convert to ndarray as it will be processed later
        data_ori = None
    else:
        df.sort_values(by=key, inplace=True)
        data_ori = df.to_records(index=False)
        if to_ndarray:
            data_ori = recarr_to_ndarray(data_ori)
    return df, data_ori


def data_manipulation_change_ratio(df, ops="None", change_ratio=0.1, to_ndarray=True):
    if os.environ["MODE"] == "full":
        if ops == "None":
            pass
        else:
            np.random.seed(0)
            max_key = np.max(df.iloc[:, 0])
            df_ori = df
            if ops == "Default":
                pass
            elif ops == "Insert":
                # rest are inserted
                num_inserted_records = int(len(df) * change_ratio)
                if "cd_education_status" in df.columns and df.shape[1] == 2:
                    # single value column high correlation case
                    df_gen_base = df_ori.iloc[:num_inserted_records].copy()
                    df_gen_base.iloc[:, 0] = np.arange(
                        max_key + 1, max_key + 1 + num_inserted_records
                    )
                    df_inserted = df_gen_base

                elif "cd_dep_college_count" in df.columns:
                    # multi value column high correlation case
                    dep_college_count_value = np.max(df["cd_dep_college_count"]) - 1
                    gen_base_df = df_ori[
                        df_ori["cd_dep_college_count"] == dep_college_count_value
                    ].copy()
                    df_inserted = None
                    for append_copy_idx in range(
                        int(num_inserted_records / len(gen_base_df) + 1)
                    ):
                        append_copy = gen_base_df.copy()
                        cd_demo_sk_copy = np.arange(
                            max_key + 1 + append_copy_idx * len(append_copy),
                            max_key + 1 + (append_copy_idx + 1) * len(append_copy),
                        )
                        append_copy.loc[:, "cd_demo_sk"] = cd_demo_sk_copy
                        if df_inserted is None:
                            df_inserted = append_copy
                        else:
                            df_inserted = pd.concat((df_inserted, append_copy), axis=0)
                    df_inserted = df_inserted.iloc[:num_inserted_records]
                else:
                    # need to generate some data to insert
                    df_inserted = df.iloc[:num_inserted_records].copy()
                    df_inserted.iloc[:, 0] = np.arange(
                        max_key + 1, max_key + 1 + len(df_inserted)
                    )
                df_len = len(df)
                ori_size = len(df)
                df = pd.concat([df, df_inserted], axis=0)
                print(
                    "[INFO] INSERT Operation: Origin Size: {}, Insert DF size: {}, After INSERTED Size: {}".format(
                        df_len, len(df_inserted), len(df)
                    )
                )

            elif ops == "Update":
                num_updated_records = int(len(df) * change_ratio)
                update_index = np.zeros(len(df), dtype=bool)
                update_index_value = np.random.choice(
                    len(df), num_updated_records, replace=False
                )
                update_index[update_index_value] = True

                df_updated = df.iloc[:num_updated_records].copy()
                df_updated.iloc[:, 0] = update_index_value

                df.iloc[update_index, :] = df_updated
                print(
                    "[INFO] Update Operation: Origin Size: {}, Update DF size: {}".format(
                        len(df), len(df_updated)
                    )
                )
            elif ops == "Delete":
                num_deleted_records = int(len(df) * change_ratio)
                delete_index = np.zeros(len(df), dtype=bool)
                delete_index[
                    np.random.choice(len(df), num_deleted_records, replace=False)
                ] = True
                ori_size = len(df)
                df = df[~delete_index]
                print(
                    "[INFO] Delete Operation: Origin Size: {}, # To Delete: {}, # After Dete: {}".format(
                        ori_size, num_deleted_records, len(df)
                    )
                )
    df = change_df_int64_to_int32(df)
    key = df.columns[0]
    df.sort_values(by=key, inplace=True)
    data_ori = df.to_records(index=False)
    if to_ndarray:
        data_ori = recarr_to_ndarray(data_ori)
    return df, data_ori


def change_df_int64_to_int32(df):
    for col in df.columns:
        if df[col].dtypes == np.int64:
            df[col] = df[col].astype(np.int32)
    return df


def generate_synthetic_data(
    df_tpch_order, df_tpch_lineitem, df_tpcds_customer_demographics, size=100
):
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
    size = size * 1024 * 1024
    # generate single column low correlation
    # single column low correlation is generated based on the TPC-H order table
    df = df_tpch_order.copy()
    df.reset_index(inplace=True)
    df = df[["index", "ORDERSTATUS"]]
    df = change_df_int64_to_int32(df)
    data_in_recarray = df.to_records(index=False)
    df_single_column_low_correlation = df[
        : int(np.floor(size / data_in_recarray[0].nbytes))
    ]

    # generate multi column low correlation
    # multi column low correlation is generated based on the TPC-H lineitem table
    df = df_tpch_lineitem.copy()
    df.reset_index(inplace=True)
    df = df[
        [
            "index",
            "LINENUMBER",
            "DISCOUNT",
            "TAX",
            "RETURNFLAG",
            "LINESTATUS",
            "SHIPINSTRUCT",
            "SHIPMODE",
        ]
    ]
    df = change_df_int64_to_int32(df)
    data_in_recarray = df.to_records(index=False)
    df_multi_column_low_correlation = df[
        : int(np.floor(size / data_in_recarray[0].nbytes))
    ]

    # generate single column high correlation
    # single column high correlation is generated based on the TPC-DS customer demographics table
    df = df_tpcds_customer_demographics[["CD_DEMO_SK", "CD_EDUCATION_STATUS"]].copy()
    df = change_df_int64_to_int32(df)
    df_new = df.copy()
    data_in_recarray = df.to_records(index=False)
    data_size = data_in_recarray.nbytes
    repeat_time = int(np.floor(size / data_size))
    last_id = np.max(df.iloc[:, 0])
    for append_copy_idx in range(1, repeat_time + 1):
        append_copy = df.copy()
        cd_demo_sk_copy = np.arange(
            last_id + (append_copy_idx - 1) * len(append_copy),
            last_id + append_copy_idx * len(append_copy),
        )
        append_copy.loc[:, "CD_DEMO_SK"] = cd_demo_sk_copy
        df_new = pd.concat((df_new, append_copy))
    df_single_column_high_correlation = df_new[
        : int(np.floor(size // data_in_recarray[0].nbytes))
    ]

    # generate multiple columns high correlation
    # multiple columns high correlation is generated based on the TPC-DS customer demographics table
    df = df_tpcds_customer_demographics.copy()
    df_new = df.copy()
    data_in_recarray = df.to_records(index=False)
    data_size = data_in_recarray.nbytes
    repeat_time = int(np.floor(size / data_size))
    last_id = np.max(df.iloc[:, 0])
    for append_copy_idx in range(1, repeat_time + 1):
        append_copy = df[df["cd_dep_college_count"] == 6].copy()
        cd_demo_sk_copy = np.arange(
            last_id + (append_copy_idx - 1) * len(append_copy),
            last_id + append_copy_idx * len(append_copy),
        )
        append_copy.loc[:, "cd_demo_sk"] = cd_demo_sk_copy
        append_copy.loc[:, "cd_dep_college_count"] += append_copy_idx
        df_new = pd.concat((df_new, append_copy))
    df_multi_column_high_correlation = df_new[
        : int(np.floor(size // data_in_recarray[0].nbytes))
    ]

    return (
        df_single_column_low_correlation,
        df_single_column_high_correlation,
        df_multi_column_low_correlation,
        df_multi_column_high_correlation,
    )


def create_features(x, max_len=None):
    if max_len is None:
        max_len = len(str(np.max(x)))
    # one-hot encoding for each digit
    x_features = np.zeros((len(x), max_len * 10))
    for idx in range(len(x)):
        digit_idx = max_len - 1
        for digit in str(x[idx])[::-1]:
            x_features[idx, digit_idx * 10 + int(digit)] = 1
            digit_idx -= 1
    return x_features, max_len


def generate_query(x_start, x_end, num_query=5, sample_size=1000):
    list_sample_index = []
    for query_idx in tqdm(range(num_query)):
        np.random.seed(query_idx)
        try:
            sample_index = np.random.choice(
                np.arange(x_start, x_end + 1, dtype=np.int32),
                sample_size,
                replace=False,
            ).astype(np.int32)
        except:
            print(
                "[WARN] Sample size too big, sample with replace instead, Log: start: {}, end: {}, sample_size: {}".format(
                    x_start, x_end + 1, sample_size
                )
            )
            sample_index = np.random.choice(
                np.arange(x_start, x_end + 1, dtype=np.int32), sample_size, replace=True
            ).astype(np.int32)
        list_sample_index.append(sample_index)
    return list_sample_index


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


def save_recarray_to_disk(file_name, rec_arr):
    np.save(file_name, rec_arr, allow_pickle=True)


def load_recarray_from_disk(file_name):
    return np.load(file_name, allow_pickle=True)


def save_obj_to_disk_with_pickle(file_name, data):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_obj_from_disk_with_pickle(file_name):
    with open(file_name, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_hashtable_to_disk(file_name, hashtable):
    with open(file_name, "wb") as f:
        pickle.dump(hashtable, f)


def load_hashtable_from_disk(file_name):
    with open(file_name, "rb") as f:
        hash_table = pickle.load(f)
    return hash_table


def get_size_of_file(file_name):
    return os.path.getsize(file_name)


class Timer(object):
    """A convenient class to measure the running time of a program"""

    def __init__(self):
        self.start = 0
        self.end = 0

    def tic(self):
        """Tic the start time"""
        self.start = time.perf_counter()

    def toc(self):
        """Toc the end time and return the running time

        Returns:
            float: running time (ms)
        """
        self.end = time.perf_counter()
        return (self.end - self.start) * 1000


def binary_search(
    x, val, num_record, search_larger_value=False, search_smaller_value=False
):
    low = 0
    high = num_record - 1
    while low <= high:
        mid = low + ((high - low) // 2)

        if x[mid] == val:
            return mid
        elif val < x[mid]:
            high = mid - 1
        elif val > x[mid]:
            low = mid + 1
    if search_larger_value:
        return low
    if search_smaller_value:
        return high
    else:
        return -1


def get_available_memory():
    return psutil.virtual_memory()[1]


def evict_unused_partition(decomp_block_dict, partition_hit, free_memory):
    max_try = 10
    curr_try = 0
    while get_available_memory() < free_memory:
        if curr_try > max_try:
            return
        curr_try += 1
        list_least_used_partition_id = nsmallest(
            100, partition_hit, key=partition_hit.get
        )
        for least_used_partition_id in list_least_used_partition_id:
            # least_used_partition_id = min(partition_hit, key=partition_hit.get)
            try:
                del decomp_block_dict[least_used_partition_id]
                del partition_hit[least_used_partition_id]
            except:
                continue
            # print('[DEBUG] eviction work')
        decomp_block_dict = dict(decomp_block_dict)
        # gc.collect()
        if decomp_block_dict is None:
            return dict()
    return decomp_block_dict


def get_nested_dict_size(d):
    total_size = sys.getsizeof(d)
    for value in d.values():
        if isinstance(value, dict):
            total_size += get_nested_dict_size(value)
        else:
            total_size += sys.getsizeof(value)
    return total_size


def get_sys_num_threads():
    return multiprocessing.cpu_count()


def get_current_used_memory(scale="MB"):
    memory_info = psutil.virtual_memory()
    used_memory = memory_info.used
    if scale == "KB":
        used_memory /= 1024**1
    elif scale == "MB":
        used_memory /= 1024**2
    elif scale == "GB":
        used_memory /= 1024**3
    return used_memory


def get_best_block_size_and_zstd_level(benchmark):
    """Get the best-tuned hyperparameters. The hyperparameters are hardware specific, 
    you need to run a grid-search to tune the hyperparameters.
    """

    block_size = None
    zstd_compress_level = None
    if benchmark == "uncompress":
        block_size = 1024 * 128
    elif benchmark == "zstd":
        block_size = 1024 * 128
        zstd_compress_level = 6
    elif benchmark == "hashtable":
        block_size = 1024 * 128
    elif benchmark == "hashtable_with_compression":
        block_size = 1024 * 128
        zstd_compress_level = 11
    elif benchmark == "deepmapping":
        block_size = 1024 * 1024 * 4
        zstd_compress_level = 15
    return block_size, zstd_compress_level


def get_temp_file_path():
    # Create a temporary file
    temp_file_descriptor, temp_file_path = tempfile.mkstemp()

    # Define a context manager to ensure cleanup
    class TempFileContext:
        def __enter__(self):
            return temp_file_path

        def __exit__(self, exc_type, exc_value, traceback):
            # Explicitly close and delete the temporary file
            os.close(temp_file_descriptor)
            os.remove(temp_file_path)
            # print(f'Temporary file deleted: {temp_file_path}')

    return TempFileContext()
