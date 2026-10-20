import logging
import math
import shutil
import subprocess
import sysconfig
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

logger = logging.getLogger(__name__)


class TpcgenDataGenerator:
    """Generate TPC data with the ``tpcgen-cli`` executable from the tpcgen-cli PyPI package.

    Output is written in tpcgen-cli's layout, one folder per table:
    ``<target>/<table>/<table>.<part>.<parquet|tbl|dat>``.
    """

    #: tpcgen-cli benchmark subcommand.
    BENCHMARK = ""
    #: tpcgen-cli subcommand and file extension of the native pipe-delimited format.
    NATIVE_FORMAT = ""
    #: Approximate ZSTD(1) Parquet size of each table at SF1000, in GB. Used only to
    #: choose how many parts (files) to split each table into.
    SF1000_SIZE_GB: Dict[str, float] = {}
    TARGET_FILE_SIZE_MB = 256

    def __init__(
        self,
        scale_factor: float,
        target_folder_uri: str,
        target_row_group_size_mb: Optional[int] = None,
        compression: Optional[str] = None,
        table_list: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        output_format: str = "parquet",
    ) -> None:
        if output_format not in ("parquet", "native"):
            raise ValueError("output_format must be either 'parquet' or 'native'.")
        if output_format == "native" and (target_row_group_size_mb is not None or compression is not None):
            raise ValueError("target_row_group_size_mb and compression apply only to output_format='parquet'.")
        table_list = list(table_list or self.SF1000_SIZE_GB)
        unknown_tables = sorted(set(table_list) - set(self.SF1000_SIZE_GB))
        if unknown_tables:
            raise ValueError(f"Unsupported {self.BENCHMARK} tables: {', '.join(unknown_tables)}")

        self.scale_factor = scale_factor
        self.target_folder = _local_path(target_folder_uri)
        self.target_row_group_size_mb = target_row_group_size_mb or 128
        self.compression = compression or "ZSTD(1)"
        self.table_list = table_list
        self.num_threads = num_threads
        self.output_format = output_format
        self.executable = _find_executable()
        self.parts_by_table = {
            table: max(1, math.ceil(self.SF1000_SIZE_GB[table] * scale_factor / 1000 * 1024 / self.TARGET_FILE_SIZE_MB))
            for table in table_list
        }

    def run(self) -> None:
        self.target_folder.mkdir(parents=True, exist_ok=True)
        tables_by_parts = defaultdict(list)
        for table in self.table_list:
            # tpcgen-cli skips files that already exist; clear any previous output.
            shutil.rmtree(self.target_folder / table, ignore_errors=True)
            tables_by_parts[self.parts_by_table[table]].append(table)
        for parts, tables in tables_by_parts.items():
            logger.info("Generating %s tables %s with %d part(s).", self.BENCHMARK, ", ".join(tables), parts)
            _run(self._command(tables, parts))

    def _command(self, tables: List[str], parts: int) -> List[str]:
        command = [
            self.executable,
            self.BENCHMARK,
            "parquet" if self.output_format == "parquet" else self.NATIVE_FORMAT,
            "--scale-factor",
            str(self.scale_factor),
            "--output-dir",
            str(self.target_folder),
            "--tables",
            ",".join(tables),
            "--parts",
            str(parts),
            "--no-progress",
        ]
        if self.output_format == "parquet":
            command += ["--compression", self.compression, "--row-group-bytes", f"{self.target_row_group_size_mb}MiB"]
        if self.num_threads is not None:
            command += ["--num-threads", str(self.num_threads)]
        return command


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(f"//{parsed.netloc}{parsed.path}")).resolve()
    if "://" in uri:
        raise ValueError(f"tpcgen-cli writes to local or mounted paths only, not {parsed.scheme}:// URIs.")
    return Path(uri).expanduser().resolve()


def _find_executable() -> str:
    # Prefer the running interpreter's scripts directory: notebook kernels often
    # do not have the environment's bin directory on PATH.
    executable = shutil.which("tpcgen-cli", path=sysconfig.get_path("scripts")) or shutil.which("tpcgen-cli")
    if executable is None:
        raise ImportError(
            "tpcgen-cli is not installed. Install it with `pip install lakebench[tpch_datagen]` or "
            "`pip install lakebench[tpcds_datagen]`. TPC-DS can alternatively use backend='duckdb' "
            "with lakebench[tpcds_duckdb_datagen] on Python 3.10 or newer."
        )
    return executable


def _run(command: List[str]) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        details = [f"tpcgen-cli failed with exit code {exc.returncode}: {' '.join(command)}"]
        if exc.stdout:
            details.append(f"stdout:\n{exc.stdout.rstrip()}")
        if exc.stderr:
            details.append(f"stderr:\n{exc.stderr.rstrip()}")
            if "Failed to rename" in exc.stderr:
                details.append(
                    "Mounted filesystems can fail under concurrent file renames. Retry with a lower "
                    "num_threads value, such as num_threads=8 or num_threads=16."
                )
        raise RuntimeError("\n".join(details)) from exc
    for output in (result.stdout, result.stderr):
        if output:
            logger.info(output.rstrip())
