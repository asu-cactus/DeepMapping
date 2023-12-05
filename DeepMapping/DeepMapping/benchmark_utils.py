from DeepMapping import (
    dgpe_compression,
    byte_dictionary_compression,
    delta_compression,
    lzo_compression,
    zstd_compression,
    uncompress,
    rle_compression,
    deepmapping,
    hashtable,
    hashtable_with_compression,
)


def benchmark_handler(benchmark, bench_type="single"):
    """Return the corresponding function call"""
    if bench_type == "single":
        if benchmark == "uncompress":
            return uncompress.measure_latency
        elif benchmark == "dgpe":
            return dgpe_compression.measure_latency
        elif benchmark == "delta":
            return delta_compression.measure_latency
        elif benchmark == "byte_dictionary":
            return byte_dictionary_compression.measure_latency
        elif benchmark == "lzo":
            return lzo_compression.measure_latency
        elif benchmark == "zstd":
            return zstd_compression.measure_latency
        elif benchmark == "rle":
            return rle_compression.measure_latency
        elif benchmark == "hashtable":
            return hashtable.measure_latency
        elif benchmark == "hashtable_with_compression":
            return hashtable_with_compression.measure_latency
        elif benchmark == "deepmapping":
            return deepmapping.measure_latency_any
        else:
            raise ValueError("NON-EXIST benchmark")
    elif bench_type == "update":
        if benchmark == "uncompress":
            return uncompress.measure_latency_data_update
        elif benchmark == "zstd":
            return zstd_compression.measure_latency_data_update
        elif benchmark == "hashtable":
            return hashtable.measure_latency_data_update
        elif benchmark == "hashtable_with_compression":
            return hashtable_with_compression.measure_latency_data_update
        elif benchmark == "deepmapping":
            return deepmapping.measure_latency_data_update
        else:
            raise ValueError("NON-EXIST benchmark")

    else:
        raise ValueError("Non supported bench_type")
