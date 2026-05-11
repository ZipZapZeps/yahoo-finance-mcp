"""Compile server.py to verified bytecode (.pyc) and report results."""

from __future__ import annotations

import compileall
import py_compile
import sys
from pathlib import Path


_ROOT = Path(__file__).parent
_TARGETS = ["server.py", "compile_server.py"]


def main() -> None:
    ok = True
    for name in _TARGETS:
        path = _ROOT / name
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"[OK]   {path.name}")
        except py_compile.PyCompileError as exc:
            print(f"[FAIL] {path.name}: {exc}", file=sys.stderr)
            ok = False

    # Compile all Python files in the project root with optimization level 2
    compileall.compile_dir(
        str(_ROOT),
        quiet=1,
        legacy=False,
        optimize=2,
        maxlevels=1,
    )
    print("Bytecode compilation complete.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
