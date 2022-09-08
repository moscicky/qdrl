import argparse


def get_args():
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--config-file-path',
        type=str,
        default=None,
        required=True
    )

    args_parser.add_argument(
        '--commit-hash',
        type=str,
        default=None,
        required=False
    )

    return args_parser.parse_args()
