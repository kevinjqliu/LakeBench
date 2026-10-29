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
    assert _arg(command, "--parts") == "6"
    assert _arg(command, "--compression") == "ZSTD(1)"
    assert _arg(command, "--row-group-bytes") == "128MiB"
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
    assert _arg(command, "--row-group-bytes") == "64MiB"


def test_parts_scale_with_table_size(fake_executable, tmp_path):
    generator = TPCHDataGenerator(scale_factor=1000, target_folder_uri=str(tmp_path))

    # 145.7 GB of lineitem at SF1000 in 256 MiB files.
    assert generator.parts_by_table["lineitem"] == 583
    assert generator.parts_by_table["nation"] == 1


def test_tables_with_equal_part_counts_share_one_command(tmp_path, commands):
    TPCHDataGenerator(scale_factor=1, target_folder_uri=str(tmp_path)).run()

    tables = sorted(table for command in commands() for table in _arg(command, "--tables").split(","))
    assert tables == sorted(TPCHDataGenerator.SF1000_SIZE_GB)
    assert len(commands()) == len(set(TPCHDataGenerator(1, str(tmp_path)).parts_by_table.values()))


def test_run_clears_previous_table_output(tmp_path, commands):
    stale = tmp_path / "region" / "region.2.parquet"
    stale.parent.mkdir()
    stale.write_bytes(b"old")

    TPCHDataGenerator(scale_factor=1, target_folder_uri=str(tmp_path), table_list=["region"]).run()

    assert not stale.exists()


def test_cleanup_failure_stops_generation(monkeypatch, tmp_path, commands):
    (tmp_path / "region").mkdir()
    monkeypatch.setattr(_tpcgen.shutil, "rmtree", Mock(side_effect=PermissionError("locked")))

    with pytest.raises(PermissionError):
        TPCHDataGenerator(scale_factor=1, target_folder_uri=str(tmp_path), table_list=["region"]).run()
    assert commands() == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"output_format": "csv"}, "output_format"),
        ({"table_list": []}, "at least one table"),
        ({"output_format": "native", "compression": "SNAPPY"}, "apply only to output_format='parquet'"),
        ({"output_format": "native", "target_row_group_size_mb": 64}, "apply only to output_format='parquet'"),
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
        ({"backend": "duckdb", "output_format": "native"}, "supported only by backend='rust'"),
    ],
)
def test_tpcds_backend_options_are_validated(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        TPCDSDataGenerator(1, str(tmp_path), **kwargs)
