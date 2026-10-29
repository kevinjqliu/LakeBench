import logging
import math
import shutil
import subprocess
import sysconfig
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import url2pathname

logger = logging.getLogger(__name__)


class TpcgenDataGenerator:
    """Generate TPC data with the ``tpcgen-cli`` executable from the tpcgen-cli PyPI package.

    Output is written in tpcgen-cli's layout, one folder per table:
    ``<target>/<table>/<table>.<part>.<parquet|tbl|dat>``.

    Each table is split into files and row groups sized from per-table measurements,
    so that on-disk sizes track the targets regardless of how well a table compresses.
    """

    #: tpcgen-cli benchmark subcommand.
    BENCHMARK = ""
    #: tpcgen-cli subcommand and file extension of the native pipe-delimited format.
    NATIVE_FORMAT = ""
    #: Benchmark-specific tpcgen-cli arguments, passed to every command.
    BENCHMARK_ARGUMENTS: List[str] = []
    #: ZSTD(1) Parquet size of each table at SF1000, in GB.
    SF1000_SIZE_GB: Dict[str, float] = {}
    #: Ratio of uncompressed to on-disk Parquet bytes per table, measured from SF10 output.
    ZSTD1_COMPRESSION_FACTOR: Dict[str, float] = {}
    SNAPPY_COMPRESSION_FACTOR: Dict[str, float] = {}
    #: Ratio of native pipe-delimited bytes to uncompressed Parquet bytes, measured per
    #: table at SF1. Both sides are measured at the same scale factor, so the ratio is a
    #: property of the encoding and holds at any scale factor.
    NATIVE_SIZE_FACTOR: Dict[str, float] = {}
    #: (table size upper bound in GiB, target file size in MiB). Larger tables use 1 GiB files.
    TARGET_FILE_SIZE_MAP = [(10, 128), (1024, 256), (5120, 512), (10240, 1024)]
    #: Tables below the first threshold are split into files of at least this fraction
    #: of a row group, so small tables are not fragmented into tiny files.
    MINIMUM_FILE_SIZE_RATIO = 0.5
    MINIMUM_FILE_SIZE_ESTIMATION_HEADROOM = 1.10
    #: Margin for tpcgen-cli's estimated bytes-per-row model when converting the
    #: on-disk row-group target to the uncompressed ``--row-group-bytes`` value.
    ROW_GROUP_ESTIMATION_HEADROOM = 1.05

    def __init__(
        self,
        scale_factor: float,
        target_folder_uri: str,
        target_row_group_size_mb: Optional[int] = None,
        compression: Optional[str] = None,
        table_list: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        compression_factor: Optional[float] = None,
        output_format: str = "parquet",
    ) -> None:
        if output_format not in ("parquet", "native"):
            raise ValueError("output_format must be either 'parquet' or 'native'.")
        if output_format == "native" and (
            target_row_group_size_mb is not None or compression is not None or compression_factor is not None
        ):
            raise ValueError(
                "target_row_group_size_mb, compression, and compression_factor apply only to output_format='parquet'."
            )
        if scale_factor <= 0:
            raise ValueError("scale_factor must be greater than zero.")
        if target_row_group_size_mb is not None and target_row_group_size_mb <= 0:
            raise ValueError("target_row_group_size_mb must be greater than zero.")
        if compression_factor is not None and compression_factor < 1:
            raise ValueError("compression_factor must be greater than or equal to one.")
        table_list = list(table_list or self.SF1000_SIZE_GB)
        unknown_tables = sorted(set(table_list) - set(self.SF1000_SIZE_GB))
        if unknown_tables:
            raise ValueError(f"Unsupported {self.BENCHMARK} tables: {', '.join(unknown_tables)}")

        self.scale_factor = scale_factor
        self.target_folder = _local_path(target_folder_uri)
        self.target_row_group_size_mb = target_row_group_size_mb or 128
        # Native text has no column compression, so its size is estimated from uncompressed Parquet.
        self.compression = (compression or "ZSTD(1)").upper() if output_format == "parquet" else "UNCOMPRESSED"
        self.table_list = table_list
        self.num_threads = num_threads
        self.output_format = output_format
        self.executable = _find_executable()
        self.compression_factor_by_table = {
            table: self._compression_factor(table, compression_factor) for table in table_list
        }
        self.parts_by_table = {table: self._parts(table) for table in table_list}

    def run(self) -> None:
        self.target_folder.mkdir(parents=True, exist_ok=True)
        tables_by_settings = defaultdict(list)
        for table in self.table_list:
            # tpcgen-cli skips files that already exist; clear any previous output.
            shutil.rmtree(self.target_folder / table, ignore_errors=True)
            tables_by_settings[self._command_group(table)].append(table)
        for (parts, compression_factor), tables in tables_by_settings.items():
            if compression_factor is None:
                # Using the group's largest factor keeps every file in a single row group.
                compression_factor = max(self.compression_factor_by_table[table] for table in tables)
            logger.info("Generating %s tables %s with %d part(s).", self.BENCHMARK, ", ".join(tables), parts)
            _run(self._command(tables, parts, compression_factor))

    def _command_group(self, table: str) -> Tuple[int, Optional[float]]:
        """Return the (parts, compression factor) a table must share a tpcgen-cli command with.

        ``--row-group-bytes`` applies to a whole command, so tables whose files span several
        row groups need their own factor. A file that fits in one row group is unaffected by
        the setting, so those tables share a command; generating many small tables in one
        command keeps every thread busy instead of running them one after another.
        """
        parts = self.parts_by_table[table]
        file_size_mb = self._estimated_size_gib(table) * 1024 / parts
        if file_size_mb * self.ROW_GROUP_ESTIMATION_HEADROOM <= self.target_row_group_size_mb:
            return parts, None
        return parts, self.compression_factor_by_table[table]

    def _command(self, tables: List[str], parts: int, compression_factor: float) -> List[str]:
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
            *self.BENCHMARK_ARGUMENTS,
            "--parts",
            str(parts),
            "--no-progress",
        ]
        if self.output_format == "parquet":
            row_group_bytes = round(
                self.target_row_group_size_mb * compression_factor * self.ROW_GROUP_ESTIMATION_HEADROOM * 1024 * 1024
            )
            command += ["--compression", self.compression, "--row-group-bytes", str(row_group_bytes)]
        if self.num_threads is not None:
            command += ["--num-threads", str(self.num_threads)]
        return command

    def _compression_factor(self, table: str, compression_factor: Optional[float]) -> float:
        if compression_factor is not None:
            return compression_factor
        if self.compression == "UNCOMPRESSED":
            return 1.0
        # Every ZSTD level is planned with the ZSTD(1) measurements.
        if self.compression.startswith("ZSTD("):
            return self.ZSTD1_COMPRESSION_FACTOR[table]
        if self.compression == "SNAPPY":
            return self.SNAPPY_COMPRESSION_FACTOR[table]
        raise ValueError(
            f"compression_factor is required for {self.compression}; "
            "measured defaults are available only for ZSTD(N) and SNAPPY."
        )

    def _parts(self, table: str) -> int:
        size_gib = self._estimated_size_gib(table)
        if size_gib < self.TARGET_FILE_SIZE_MAP[0][0]:
            minimum_file_size_mb = (
                self.target_row_group_size_mb
                * self.MINIMUM_FILE_SIZE_RATIO
                * self.MINIMUM_FILE_SIZE_ESTIMATION_HEADROOM
            )
            return max(1, math.floor(size_gib * 1024 / minimum_file_size_mb))
        target_file_size_mb = next(
            (size_mb for threshold_gib, size_mb in self.TARGET_FILE_SIZE_MAP if size_gib < threshold_gib), 1024
        )
        return max(1, math.ceil(size_gib * 1024 / target_file_size_mb))

    def _estimated_size_gib(self, table: str) -> float:
        """Estimated on-disk size of a table in the requested compression or native format."""
        size_gib = (
            self._zstd1_size_gib(table) * self.ZSTD1_COMPRESSION_FACTOR[table] / self.compression_factor_by_table[table]
        )
        if self.output_format == "native":
            size_gib *= self.NATIVE_SIZE_FACTOR[table]
        return size_gib

    def _zstd1_size_gib(self, table: str) -> float:
        """Estimated ZSTD(1) Parquet size of a table at the requested scale factor.

        Assumes linear scaling; benchmarks with fixed-size tables override this.
        """
        return self.SF1000_SIZE_GB[table] * self.scale_factor / 1000


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
