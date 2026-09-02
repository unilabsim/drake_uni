from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "src/drake_uni/compiled/drake_env_pool.cc"
OUTPUT_DIR = REPO_ROOT / "src/drake_uni/compiled"


def _pkg_config_flags(package: str, option: str) -> list[str]:
    try:
        result = subprocess.run(
            ["pkg-config", option, package],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return result.stdout.split()


def _include_flags(env_name: str, fallback: Path | None = None) -> list[str]:
    """Return include flags from an override, falling back to a known path."""

    values = [item for item in os.environ.get(env_name, "").split(os.pathsep) if item]
    if not values and fallback is not None and fallback.is_dir():
        values = [str(fallback)]
    return [f"-I{value}" for value in values]


def _library_flags(env_name: str) -> list[str]:
    """Return ``-L`` flags for libraries outside the system search path."""

    values = [item for item in os.environ.get(env_name, "").split(os.pathsep) if item]
    return [f"-L{value}" for value in values]


def build_command(drake_home: Path, output: Path) -> list[str]:
    python_include = Path(sysconfig.get_paths()["include"])
    pybind_include = drake_home / "include/pybind11"
    eigen_flags = _pkg_config_flags("eigen3", "--cflags")
    if not eigen_flags:
        eigen_flags = _include_flags("EIGEN3_INCLUDE_DIR", drake_home / "include/eigen3")
    fmt_include_flags = _include_flags("FMT_INCLUDE_DIR")
    fmt_flags = _pkg_config_flags("fmt", "--cflags")
    if fmt_include_flags:
        fmt_flags.extend(fmt_include_flags)
    include_flags = [
        f"-I{python_include}",
        f"-I{drake_home / 'include'}",
        f"-I{pybind_include}",
        *eigen_flags,
        *fmt_flags,
    ]
    lib_flags = [
        *_pkg_config_flags("fmt", "--libs"),
        *_library_flags("FMT_LIB_DIR"),
    ]
    if not any(flag == "-lfmt" for flag in lib_flags):
        lib_flags.append("-lfmt")
    lib_dir = drake_home / "lib"
    python_link_flags = []
    if platform.system() == "Darwin":
        python_link_flags = ["-undefined", "dynamic_lookup"]
    compiler = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++"
    # CPython commonly reports ``g++ -pthread`` as its CXX value.  Keep
    # user-supplied compiler flags while passing the executable and arguments
    # separately to subprocess rather than treating the whole string as a
    # filename.
    compiler_command = shlex.split(compiler)
    return [
        *compiler_command,
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-shared",
        *python_link_flags,
        *include_flags,
        str(SOURCE),
        f"-L{lib_dir}",
        "-ldrake",
        *lib_flags,
        "-Wl,-rpath," + str(lib_dir),
        "-o",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drake-home",
        default=os.environ.get("DRAKE_HOME"),
        help="Drake C++ install prefix containing include/ and lib/libdrake.so.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.drake_home:
        raise SystemExit("Set DRAKE_HOME or pass --drake-home to the Drake C++ install prefix.")
    drake_home = Path(args.drake_home).expanduser().resolve()
    if not (drake_home / "include/drake").is_dir():
        raise SystemExit(f"Drake include directory not found under {drake_home}")
    if not (drake_home / "lib/libdrake.so").exists():
        raise SystemExit(f"Drake shared library not found under {drake_home}")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise SystemExit("Could not determine Python extension suffix")
    output = OUTPUT_DIR / ("_drake_env_pool" + ext_suffix)
    command = build_command(drake_home, output)
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
