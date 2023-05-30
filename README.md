# DeepMapping: The Case for Learned Data Mapping for Compression and Efficient Query Processing

Resources for SIGMOD 2024 Submission

<!-- TOC -->

- [DeepMapping: The Case for Learned Data Mapping for Compression and Efficient Query Processing](#deepmapping-the-case-for-learned-data-mapping-for-compression-and-efficient-query-processing)
    - [Set-Up](#set-up)
    - [Dataset](#dataset)
    - [Model Searching](#model-searching)
    - [Benchmark](#benchmark)
        - [Task: Data Query](#task-data-query)
        - [Task: Data Manipulation](#task-data-manipulation)
    - [Supplement Material](#supplement-material)
        - [Comparison of end-end latency using hashing and binary search](#comparison-of-end-end-latency-using-hashing-and-binary-search)
        - [Comparison of end-end latency for running model in CPU/GPU](#comparison-of-end-end-latency-for-running-model-in-cpugpu)
        - [Comparison of tunning the partition size](#comparison-of-tunning-the-partition-size)

<!-- /TOC -->

## Set-Up

1. Please install the required dependencies via `pip install -r requirements.txt`

2. DeepMapping is wrapped up as a Python library, please run the following command to install it.

    ```
    cd DeepMapping
    pip install -e ./
    ```

3. We wrapped up the feature extraction as a C-based function for better performance. Run the following command to compile it as a shared library.

    ```
    cc -fPIC -shared -o shared_utils.so shared_utils.c
    ```

## Dataset

Our experiments covered synthetic datasets, low/high correlation datasets with different scales(100MB, 1GB, 10GB), and TPC-H, TPC-DS benchmark datasets with scale factors as 1 and 10. We removed all string/continuous columns and uploaded our pre-generated datasets to [**HERE**](https://mega.nz/file/aUREBDQI#vW-rUQOTOr0B7uN9XhcOFXd2dqfe5yA18-Mk3xn-Dvc):

[**DATA LINK: https://mega.nz/file/aUREBDQI#vW-rUQOTOr0B7uN9XhcOFXd2dqfe5yA18-Mk3xn-Dvc**](https://mega.nz/file/aUREBDQI#vW-rUQOTOr0B7uN9XhcOFXd2dqfe5yA18-Mk3xn-Dvc)

After download it, please unzip it to the **root** folder of this GitHub repository. Then, you will see a **dataset**  folder here.

List of datasets:

- TPC-H (S1/S10): `customer`, `lineitem`, `orders`, `part`, `supplier`.
- TPC-DS (S1/S10): `catalog_page`, `catalog_returns`, `catalog_sales`,`customer_address`, `customer_demographics`, `customer`, `item`, `store_returns`, `web_returns`.
- Synthetic Dataset (100MB, 1GB, 10GB): `single_column_low_correlation`, `single_column_high_correlation`, `multiple_column_low_correlation`, `multiple_column_high_correlation`.


## Model Searching

1. Please run `python run_search_model.py` to perform a NAS with given dataset. You can configure the NAS by editing the **run_search_model.py** correspondingly. The searched result will be printout.

2. Modify the `SEARCH_MODEL_STRUCTURE` in `run_train_searched_model.py` with the output from step 1. And then run `python run_train_searched_model.py` to train a model.

## Benchmark 

We provided some example models for the following 2 tasks. Please go [**HERE**](https://mega.nz/file/SdYWHAzZ#AAuYAz_-UmHXWUixHGOzzBJN0NTmwY6N66da3UyRS9s) to download:

[**MODEL LINK: https://mega.nz/file/SdYWHAzZ#AAuYAz_-UmHXWUixHGOzzBJN0NTmwY6N66da3UyRS9s**](https://mega.nz/file/SdYWHAzZ#AAuYAz_-UmHXWUixHGOzzBJN0NTmwY6N66da3UyRS9s)

After download it, please unzip it to the **root** folder of this GitHub repository. Then, you will see a **models**  folder here.




### Task: Data Query

These experiments measured overall storage overhead and end-end query latency for benchmark datasets, i.e. TPC-H and TPC-DS. 
Run `python run_benchmark_data_query.py` to benchmark. To benchmark with different dataset, you should modify the file correspondingly by following the instructions provided in the python file.

### Task: Data Manipulation

These experiments measured overall storage overhead and end-end query latency for synthetic dataset with data manipulation, i.e. INSERT/UPDATE/DELETE. Run `python run_benchmark_data_manipulation.py` to benchmark it. To benchmark with different dataset, you should modify the file correspondingly by following the instructions provided in the python file.

## Supplement Material

### Comparison of end-end latency using hashing and binary search

|                  |                          |             | Binary Search |            |      Hashing     |             |            |
|:----------------:|:------------------------:|:-----------:|:-------------:|:----------:|:----------------:|:-----------:|:----------:|
|       Table      | Number of Query Data (B) |    Method   |  Lookup Time  | Total Time | Build Index Time | Lookup Time | Total Time |
| TPCH-S1/customer |           1,000          |  Uncompress |     65.72     |    79.28   |      470.83      |     6.91    |   491.53   |
|                  |           1,000          |  Z-Standard |     65.94     |    83.12   |      463.97      |     6.66    |   487.91   |
|                  |           1,000          | DeepMapping |     37.32     |    68.87   |       46.09      |     4.76    |    82.88   |
| TPCH-S1/lineitem |           1,000          |  Uncompress |     74.22     |   244.22   |     16,924.19    |    12.68    |  17,096.75 |
|                  |           1,000          |  Z-Standard |     73.28     |   643.76   |     16,969.66    |    12.72    |  17,568.03 |
|                  |           1,000          | DeepMapping |     42.67     |   496.54   |     2,274.11     |     6.33    |  2,708.15  |
|  TPCH-S1/orders  |           1,000          |  Uncompress |     60.81     |    83.82   |     4,498.52     |     5.21    |  4,532.56  |
|                  |           1,000          |  Z-Standard |     60.94     |   126.41   |     4,481.43     |     5.12    |  4,558.40  |
|                  |           1,000          | DeepMapping |      8.76     |    66.51   |      545.31      |     1.52    |   611.88   |
|   TPCH-S1/part   |           1,000          |  Uncompress |     44.81     |    61.16   |      614.14      |     7.69    |   637.64   |
|                  |           1,000          |  Z-Standard |     44.75     |    68.63   |      618.00      |     7.71    |   649.57   |
|                  |           1,000          | DeepMapping |     36.28     |    76.22   |       63.59      |     4.96    |   111.11   |
| TPCH-S1/supplier |           1,000          |  Uncompress |     61.44     |    73.34   |       30.56      |     4.80    |    46.93   |
|                  |           1,000          |  Z-Standard |     61.60     |    73.49   |       28.18      |     4.82    |    44.61   |
|                  |           1,000          | DeepMapping |     31.87     |    55.06   |       2.92       |     3.99    |    29.66   |


### Comparison of end-end latency for running model in CPU/GPU

|       Tables      | Number of Query Data (B) |       Method      | NN Inference Time | Total Time |
|:-----------------:|:------------------------:|:-----------------:|:-----------------:|:----------:|
| TPCH-S10/customer |          100,000         |     Uncompress    |                   |   7782.68  |
|                   |          100,000         | DeepMapping (GPU) |       110.84      |  5,454.34  |
|                   |          100,000         | DeepMapping (CPU) |       663.30      |  6,058.63  |
| TPCH-S10/lineitem |          100,000         |     Uncompress    |                   |  11102.67  |
|                   |          100,000         | DeepMapping (GPU) |       294.86      |  10,511.73 |
|                   |          100,000         | DeepMapping (CPU) |      2,909.44     |  12,791.68 |
|  TPCH-S10/orders  |          100,000         |     Uncompress    |                   |   7666.93  |
|                   |          100,000         | DeepMapping (GPU) |       81.38       |  2,351.74  |
|                   |          100,000         | DeepMapping (CPU) |       403.41      |  2,707.00  |
|   TPCH-S10/part   |          100,000         |     Uncompress    |                   |   5960.09  |
|                   |          100,000         | DeepMapping (GPU) |       238.95      |  5,686.15  |
|                   |          100,000         | DeepMapping (CPU) |       310.98      |  5,804.16  |
| TPCH-S10/supplier |          100,000         |     Uncompress    |                   |   7793.17  |
|                   |          100,000         | DeepMapping (GPU) |       56.59       |  5,097.88  |
|                   |          100,000         | DeepMapping (CPU) |       75.77       |  5,153.22  |
### Comparison of tunning the partition size

Experiments results are measured on TPC-H, SF=10, `orders` table, B=100,000

| Partition Size |    Method   | Compressed Size | Load (Decompression) Time | Look-Up Time | Total Time |
|:--------------:|:-----------:|:---------------:|:-------------------------:|:------------:|:----------:|
|     128 KB     |  Uncompress |      343.67     |           697.19          |    4572.13   |   6841.63  |
|                |  Z-Standard |      42.68      |          1245.57          |    4282.95   |   7089.68  |
|                | DeepMapping |      34.42      |           546.76          |    792.16    |   2492.61  |
|     256 KB     |  Uncompress |      343.5      |           505.85          |    4583.34   |   6646.95  |
|                |  Z-Standard |      42.17      |          1053.05          |    4501.56   |   7120.86  |
|                | DeepMapping |       34.5      |           478.98          |    827.44    |   2453.57  |
|     512 KB     |  Uncompress |      343.41     |           419.23          |    4818.57   |   6784.54  |
|                |  Z-Standard |      41.82      |           932.45          |    4753.03   |   7230.98  |
|                | DeepMapping |      34.73      |           433.73          |    863.55    |   2435.87  |
|      1 MB      |  Uncompress |      343.37     |           369.81          |    4912.53   |   6836.51  |
|                |  Z-Standard |      41.51      |           878.25          |    4887.92   |   7312.56  |
|                | DeepMapping |      34.46      |           419.45          |     904.3    |   2462.09  |
|      2 MB      |  Uncompress |      343.34     |           357.78          |    5151.33   |   7041.75  |
|                |  Z-Standard |       41.4      |           834.38          |    5058.49   |   7418.55  |
|                | DeepMapping |      34.61      |           413.97          |    947.68    |   2502.53  |
|      4 MB      |  Uncompress |      343.33     |           331.85          |    5322.3    |   7187.15  |
|                |  Z-Standard |      41.34      |           847.77          |    5351.77   |   7727.56  |
|                | DeepMapping |      34.66      |           412.57          |    990.98    |   2544.98  |
|      8 MB      |  Uncompress |      343.33     |           341.84          |    5602.07   |   7484.76  |
|                |  Z-Standard |      41.57      |           839.85          |    5582.87   |   7948.59  |
|                | DeepMapping |      34.67      |           411.99          |    1027.16   |   2576.42  |
|      16 MB     |  Uncompress |      343.33     |           342.31          |    5839.79   |   7704.64  |
|                |  Z-Standard |      41.19      |           824.81          |    5781.69   |   8125.71  |
|                | DeepMapping |      34.67      |           439.3           |    1069.7    |   2646.52  |
|      32 MB     |  Uncompress |      343.32     |           341.2           |    6070.11   |   7947.69  |
|                |  Z-Standard |      41.37      |           840.95          |    6036.75   |   8400.84  |
|                | DeepMapping |      34.67      |           449.65          |    1107.8    |   2694.4   |