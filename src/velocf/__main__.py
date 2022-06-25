"""Main package entry point."""

import sys

from velocf.cli import velocf


def run() -> None:
    velocf(sys.argv[1:])


if __name__ == "__main__":
    run()
