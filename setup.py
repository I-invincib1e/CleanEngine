#!/usr/bin/env python3
"""
Bootstrap installation script.

Usage:
  python setup.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    req = PROJECT_ROOT / 'requirements.txt'
    if req.exists():
        code = run([sys.executable, '-m', 'pip', 'install', '-r', str(req)])
        if code != 0:
            return code
    else:
        print('requirements.txt not found; skipping dependency installation')
    print('Setup complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
