# DeepMapping: Learned Data Mapping for Compression and Efficient Query Processing

Resources for VLDB 2024 Submission

<!-- TOC -->

- [DeepMapping: Learned Data Mapping for Compression and Efficient Query Processing](#deepmapping-learned-data-mapping-for-compression-and-efficient-query-processing)
  - [Dataset](#dataset)
  - [Model Searching](#model-searching)
  - [Benchmark](#benchmark)
    - [Task: Data Query](#task-data-query)
    - [Task: Data Manipulation](#task-data-manipulation)

<!-- /TOC -->
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

Our experiments covered synthetic datasets, low/high correlation datasets with different scales(100MB, 1GB, 10GB), and TPC-H, TPC-DS benchmark datasets with scale factors as 1 and 10. We removed all string/continuous columns and uploaded our pre-generated datasets to [**HERE**](https://mega.nz/file/nNggnQzA#9Ma2v3GIrfR-3ndGNzGXsF5ZOcWtGwZKeRekUiqOnzA):

[**DATA LINK: Here**](https://mega.nz/file/nNggnQzA#9Ma2v3GIrfR-3ndGNzGXsF5ZOcWtGwZKeRekUiqOnzA)

After download it, please unzip it to the **root** folder of this GitHub repository. Then, you will see a **dataset**  folder here.

List of datasets:

- TPC-H (S1/S10): `customer`, `lineitem`, `orders`, `part`, `supplier`.
- TPC-DS (S1/S10): `catalog_page`, `catalog_returns`, `catalog_sales`,`customer_address`, `customer_demographics`, `customer`, `item`, `store_returns`, `web_returns`.
- Synthetic Dataset (100MB, 1GB, 10GB): `single_column_low_correlation`, `single_column_high_correlation`, `multiple_column_low_correlation`, `multiple_column_high_correlation`.


## Model Searching

1. Please run `python run_search_model.py` to perform a NAS with given dataset. You can configure the NAS by editing the **run_search_model.py** correspondingly. The searched result will be printout.

2. Modify the `SEARCH_MODEL_STRUCTURE` in `run_train_searched_model.py` with the output from step 1. And then run `python run_train_searched_model.py` to train a model.

## Benchmark 

We provided some example models for the following 2 tasks. Please go [**HERE**](https://mega.nz/file/icxG1JaL#cuC5C4_PxQ1FsgSUmswfaXyzCaatOwx9n_b9F_-IDnU) to download:

[**MODEL LINK: Here**](https://mega.nz/file/icxG1JaL#cuC5C4_PxQ1FsgSUmswfaXyzCaatOwx9n_b9F_-IDnU)

After download it, please unzip it to the **root** folder of this GitHub repository. Then, you will see a **models**  folder here.




### Task: Data Query

These experiments measured overall storage overhead and end-end query latency for benchmark datasets, i.e. TPC-H and TPC-DS. 
Run `python run_benchmark_data_query.py` to benchmark. To benchmark with different dataset, you should modify the file correspondingly by following the instructions provided in the python file.

### Task: Data Manipulation

These experiments measured overall storage overhead and end-end query latency for synthetic dataset with data manipulation, i.e. INSERT/UPDATE/DELETE. Run `python run_benchmark_data_manipulation.py` to benchmark it. To benchmark with different dataset, you should modify the file correspondingly by following the instructions provided in the python file.
