from ._tpcgen import TpcgenDataGenerator


class TPCHDataGenerator(TpcgenDataGenerator):
    """Generate TPC-H data with ``tpcgen-cli`` (``pip install lakebench[tpch_datagen]``)."""

    BENCHMARK = "tpch"
    NATIVE_FORMAT = "tbl"
    SF10_SIZE_MIB = {
        "customer": 80.2,
        "lineitem": 1528.0,
        "nation": 0.0,
        "orders": 383.1,
        "part": 44.3,
        "partsupp": 292.0,
        "region": 0.0,
        "supplier": 5.6,
    }
