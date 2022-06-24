"""Main package entry point."""

import sys

from velocf.cli import velocf

if __name__ == "__main__":
    velocf(sys.argv[1:])
