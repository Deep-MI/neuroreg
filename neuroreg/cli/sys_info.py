import argparse

from .._sys_info import sys_info


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ``neuroreg-sys_info`` command."""
    parser = argparse.ArgumentParser(
        prog=f"{__package__.split('.')[0]}-sys_info",
        description="Display dependency and runtime information for neuroreg.",
    )
    parser.add_argument(
        "--developer",
        help="Display information for optional dependencies.",
        action="store_true",
    )
    return parser


def main(args=None) -> None:
    """Run the ``neuroreg-sys_info`` command."""
    parser = _build_parser()
    args = parser.parse_args(args)

    sys_info(developer=args.developer)


def run() -> None:
    """Run the sys_info CLI entrypoint."""
    main()


if __name__ == "__main__":
    main()
