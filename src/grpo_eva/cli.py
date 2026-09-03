"""Console-script entry point; the implementation lives in ../../main.py."""
import pathlib, runpy, sys


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.argv[0] = str(root / 'main.py')
    runpy.run_path(str(root / 'main.py'), run_name='__main__')
