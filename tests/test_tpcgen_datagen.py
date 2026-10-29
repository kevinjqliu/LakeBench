import inspect
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from lakebench.datagen import TPCDSDataGenerator, TPCHDataGenerator, _tpcgen
from lakebench.datagen.tpcds import _TPCDSTpcgenDataGenerator


@pytest.fixture
def fake_executable(monkeypatch):
    monkeypatch.setattr(_tpcgen, "_find_executable", lambda: "tpcgen-cli")
    return "tpcgen-cli"


@pytest.fixture
def commands(monkeypatch, fake_executable):
    run = Mock()
    monkeypatch.setattr(_tpcgen, "_run", run)
    return lambda: [call.args[0] for call in run.call_args_list]


def _arg(command, flag):
    return command[command.index(flag) + 1]


def test_tpch_parquet_command(tmp_path, commands):
    TPCHDataGenerator(scale_factor=10, target_folder_uri=str(tmp_path), table_list=["lineitem"]).run()

    (command,) = commands()
    assert command[:3] == ["tpcgen-cli", "tpch", "parquet"]
    assert _arg(command, "--scale-factor") == "10"
    assert _arg(command, "--output-dir") == str(tmp_path.resolve())
    assert _arg(command, "--tables") == "lineitem"
    # 1.4 GiB of lineitem at SF10 is below 10 GiB, so it is split into files of at
    # least half a row group (with headroom) rather than tiny 256 MiB-tier files.
    assert _arg(command, "--parts") == "21"
    assert _arg(command, "--compression") == "ZSTD(1)"
    # 128 MiB on disk, scaled by lineitem's measured ZSTD(1) ratio and 5% headroom.
    assert _arg(command, "--row-group-bytes") == str(round(128 * 2.465 * 1.05 * 1024 * 1024))
    assert "--num-threads" not in command


def test_tpcds_native_command(tmp_path, commands):
    TPCDSDataGenerator(
        scale_factor=1,
        target_folder_uri=str(tmp_path),
        table_list=["store_sales"],
        num_threads=4,
        output_format="native",
    ).run()

    (command,) = commands()
    assert command[1:3] == ["tpcds", "dat"]
    assert _arg(command, "--num-threads") == "4"
    assert "--compression" not in command
    assert "--row-group-bytes" not in command


def test_parquet_options_are_passed_through(tmp_path, commands):
    TPCHDataGenerator(
        scale_factor=1,
        target_folder_uri=str(tmp_path),
        table_list=["region"],
        compression="SNAPPY",
        target_row_group_size_mb=64,
    ).run()

    (command,) = commands()
    assert _arg(command, "--compression") == "SNAPPY"
    assert _arg(command, "--row-group-bytes") == str(round(64 * 1.149 * 1.05 * 1024 * 1024))


def test_parts_scale_with_table_size(fake_executable, tmp_path):
    generator = TPCHDataGenerator(scale_factor=1000, target_folder_uri=str(tmp_path))

    # 145.7 GB of lineitem at SF1000 in 256 MiB files.
    assert generator.parts_by_table["lineitem"] == 583
    assert generator.parts_by_table["nation"] == 1


def test_tables_with_equal_sizing_share_one_command(tmp_path, commands):
    TPCHDataGenerator(scale_factor=1, target_folder_uri=str(tmp_path), compression_factor=2.0).run()

    tables = sorted(table for command in commands() for table in _arg(command, "--tables").split(","))
    assert tables == sorted(TPCHDataGenerator.SF1000_SIZE_GB)
    generator = TPCHDataGenerator(1, str(tmp_path), compression_factor=2.0)
    assert len(commands()) == len(set(generator.parts_by_table.values())) < len(tables)


def test_single_row_group_tables_share_a_command_with_the_largest_factor(tmp_path, commands):
    # Every file fits in one row group, so --row-group-bytes does not change the output.
    TPCHDataGenerator(scale_factor=1, target_folder_uri=str(tmp_path), table_list=["nation", "region"]).run()

    (command,) = commands()
    assert _arg(command, "--tables") == "nation,region"
    assert _arg(command, "--row-group-bytes") == str(round(128 * 2.027 * 1.05 * 1024 * 1024))


def test_multi_row_group_tables_keep_their_own_compression_factor(fake_executable, tmp_path, monkeypatch):
    generator = TPCHDataGenerator(scale_factor=1000, target_folder_uri=str(tmp_path), table_list=["lineitem", "orders"])
    # Same part count, but each file spans several row groups.
    generator.parts_by_table = {"lineitem": 10, "orders": 10}
    run = Mock()
    monkeypatch.setattr(_tpcgen, "_run", run)

    generator.run()

    row_group_bytes = sorted(int(_arg(call.args[0], "--row-group-bytes")) for call in run.call_args_list)
    assert row_group_bytes == [round(128 * factor * 1.05 * 1024 * 1024) for factor in (2.465, 3.054)]


def test_run_clears_previous_table_output(tmp_path, commands):
    stale = tmp_path / "region" / "region.2.parquet"
    stale.parent.mkdir()
    stale.write_bytes(b"old")

    TPCHDataGenerator(scale_factor=1, target_folder_uri=str(tmp_path), table_list=["region"]).run()

    assert not stale.exists()


@pytest.mark.parametrize(
    ("scale_factor", "expected_parts"),
    [
        # Under 10 GiB: files of at least half a 128 MiB row group, with 10% headroom.
        (10, 21),
        # 10 GiB to 1 TiB: 256 MiB files.
        (1000, 583),
        # 1 TiB to 5 TiB: 512 MiB files.
        (10000, 2915),
        # 5 TiB to 10 TiB: 1 GiB files.
        (50000, 7287),
    ],
)
def test_parts_follow_target_file_size_tiers(fake_executable, tmp_path, scale_factor, expected_parts):
    generator = TPCHDataGenerator(scale_factor=scale_factor, target_folder_uri=str(tmp_path), table_list=["lineitem"])

    assert generator.parts_by_table["lineitem"] == expected_parts


def test_snappy_uses_measured_compression_for_parts(fake_executable, tmp_path):
    zstd = TPCHDataGenerator(1000, str(tmp_path), table_list=["lineitem"])
    snappy = TPCHDataGenerator(1000, str(tmp_path), table_list=["lineitem"], compression="SNAPPY")

    assert snappy.compression_factor_by_table["lineitem"] == 1.866
    assert snappy._estimated_size_gib("lineitem") == pytest.approx(zstd._estimated_size_gib("lineitem") * 2.465 / 1.866)
    assert snappy.parts_by_table["lineitem"] > zstd.parts_by_table["lineitem"]


def test_all_zstd_levels_use_zstd1_measurements(fake_executable, tmp_path):
    generator = TPCHDataGenerator(1, str(tmp_path), table_list=["lineitem"], compression="zstd(9)")

    assert generator.compression == "ZSTD(9)"
    assert generator.compression_factor_by_table["lineitem"] == 2.465


def test_other_codecs_require_compression_factor(fake_executable, tmp_path):
    with pytest.raises(ValueError, match="compression_factor is required for GZIP"):
        TPCHDataGenerator(1, str(tmp_path), compression="GZIP(6)")

    generator = TPCHDataGenerator(1, str(tmp_path), compression="GZIP(6)", compression_factor=3.0)
    assert set(generator.compression_factor_by_table.values()) == {3.0}


def test_native_sizing_applies_measured_expansion_factor(fake_executable, tmp_path):
    parquet = _TPCDSTpcgenDataGenerator(1000, str(tmp_path), table_list=["store_sales"])
    native = _TPCDSTpcgenDataGenerator(1000, str(tmp_path), table_list=["store_sales"], output_format="native")

    uncompressed_gib = parquet._zstd1_size_gib("store_sales") * 1.344
    assert native._estimated_size_gib("store_sales") == pytest.approx(uncompressed_gib * 2.859)
    assert native.parts_by_table["store_sales"] > parquet.parts_by_table["store_sales"]


def test_measurements_cover_every_table():
    for generator in (TPCHDataGenerator, _TPCDSTpcgenDataGenerator):
        tables = set(generator.SF1000_SIZE_GB)
        assert set(generator.ZSTD1_COMPRESSION_FACTOR) == tables
        assert set(generator.SNAPPY_COMPRESSION_FACTOR) == tables
        assert set(generator.NATIVE_SIZE_FACTOR) == tables


def test_tpcds_dimensions_scale_by_row_count(fake_executable, tmp_path):
    sf10 = _TPCDSTpcgenDataGenerator(10, str(tmp_path))
    sf1000 = _TPCDSTpcgenDataGenerator(1000, str(tmp_path))

    # date_dim has a fixed row count, so its size does not grow with the scale factor.
    assert sf1000._zstd1_size_gib("date_dim") == pytest.approx(sf10._zstd1_size_gib("date_dim"))
    # customer grows from 500K to 12M rows between SF10 and SF1000: 24x, not 100x.
    assert sf1000._zstd1_size_gib("customer") == pytest.approx(sf10._zstd1_size_gib("customer") * 24)
    # Facts scale linearly.
    assert sf1000._zstd1_size_gib("store_sales") == pytest.approx(sf10._zstd1_size_gib("store_sales") * 100)


@pytest.mark.parametrize(
    ("table", "scale_factor", "expected_rows"),
    [
        ("customer", 10, 500000),
        ("customer", 1000, 12000000),
        ("item", 1000, 300000),
        ("warehouse", 1000, 20),
        ("date_dim", 250, 73049),
        ("inventory", 1, 11745000),
        ("inventory", 10, 133110000),
        ("inventory", 1000, 783000000),
    ],
)
def test_tpcds_expected_row_counts_match_scaling_model(table, scale_factor, expected_rows):
    assert _TPCDSTpcgenDataGenerator._expected_row_count(table, scale_factor) == expected_rows


def test_tpcds_requests_c_compatible_output(tmp_path, commands):
    TPCDSDataGenerator(scale_factor=1, target_folder_uri=str(tmp_path), table_list=["store_sales"]).run()

    (command,) = commands()
    assert _arg(command, "--compat") == "c"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"output_format": "csv"}, "output_format"),
        ({"output_format": "native", "compression": "SNAPPY"}, "apply only to output_format='parquet'"),
        ({"output_format": "native", "target_row_group_size_mb": 64}, "apply only to output_format='parquet'"),
        ({"output_format": "native", "compression_factor": 2.0}, "apply only to output_format='parquet'"),
        ({"compression_factor": 0.5}, "compression_factor must be greater than or equal to one"),
        ({"scale_factor": 0}, "scale_factor must be greater than zero"),
        ({"target_row_group_size_mb": 0}, "target_row_group_size_mb must be greater than zero"),
        ({"table_list": ["unknown"]}, "Unsupported tpch tables: unknown"),
        ({"target_folder_uri": "abfss://ws@onelake/files"}, "local or mounted paths"),
    ],
)
def test_invalid_options_are_rejected(tmp_path, fake_executable, kwargs, message):
    kwargs = {"scale_factor": 1, "target_folder_uri": str(tmp_path), **kwargs}
    with pytest.raises(ValueError, match=message):
        TPCHDataGenerator(**kwargs)


def test_file_uri_target(tmp_path, fake_executable):
    assert TPCHDataGenerator(1, tmp_path.as_uri()).target_folder == tmp_path.resolve()


def test_executable_is_found_in_interpreter_scripts_directory(monkeypatch, tmp_path):
    executable = tmp_path / ("tpcgen-cli.exe" if os.name == "nt" else "tpcgen-cli")
    executable.write_bytes(b"binary")
    executable.chmod(0o755)
    monkeypatch.setattr(_tpcgen.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setenv("PATH", "")

    assert Path(_tpcgen._find_executable()) == executable


def test_missing_executable_explains_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setattr(_tpcgen.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setenv("PATH", "")

    with pytest.raises(ImportError, match=r"lakebench\[tpch_datagen\]"):
        _tpcgen._find_executable()


def test_cli_failure_surfaces_output(monkeypatch):
    error = subprocess.CalledProcessError(2, ["tpcgen-cli"], output="partial output", stderr="bad option")
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))

    with pytest.raises(RuntimeError, match="partial output") as exc_info:
        _tpcgen._run(["tpcgen-cli"])
    assert "bad option" in str(exc_info.value)


def test_cli_rename_failure_suggests_fewer_threads(monkeypatch):
    error = subprocess.CalledProcessError(1, ["tpcgen-cli"], stderr='Failed to rename "a.inprogress" to "a"')
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))

    with pytest.raises(RuntimeError, match="num_threads=8"):
        _tpcgen._run(["tpcgen-cli"])


def test_public_generators_share_parameters():
    tpcds_parameters = list(inspect.signature(TPCDSDataGenerator).parameters)
    tpch_parameters = list(inspect.signature(TPCHDataGenerator).parameters)

    assert tpcds_parameters[:6] == tpch_parameters[:6]
    assert tpcds_parameters[-1] == tpch_parameters[-1] == "output_format"


def test_tpcds_dispatches_to_tpcgen(tmp_path, fake_executable):
    generator = TPCDSDataGenerator(1, str(tmp_path))

    assert isinstance(generator._generator, _TPCDSTpcgenDataGenerator)


def test_tpcds_dispatches_to_duckdb(monkeypatch, tmp_path):
    duckdb_generator = Mock()
    constructor = Mock(return_value=duckdb_generator)
    monkeypatch.setattr("lakebench.datagen.tpcds._TPCDSDuckDBDataGenerator", constructor)

    TPCDSDataGenerator(1, str(tmp_path), backend="duckdb").run()

    constructor.assert_called_once_with(scale_factor=1, target_folder_uri=str(tmp_path), target_row_group_size_mb=128)
    duckdb_generator.run.assert_called_once_with()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "unknown"}, "backend"),
        ({"backend": "duckdb", "compression": "SNAPPY", "num_threads": 1}, "compression, num_threads"),
        ({"backend": "duckdb", "compression_factor": 2.0}, "compression_factor"),
        ({"backend": "duckdb", "output_format": "native"}, "supported only by backend='rust'"),
    ],
)
def test_tpcds_backend_options_are_validated(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        TPCDSDataGenerator(1, str(tmp_path), **kwargs)
