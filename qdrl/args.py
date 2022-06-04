import argparse


def get_args():
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--num-epochs',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--task-id',
        type=str,
        required=True
    )

    args_parser.add_argument(
        '--run-id',
        type=str,
        required=True
    )

    args_parser.add_argument(
        '--training-data-dir',
        type=str,
        required=True,
    )

    args_parser.add_argument(
        '--training-data-file',
        type=str,
        default=None
    )

    args_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3
    )

    args_parser.add_argument(
        '--reuse-epoch',
        action='store_true',
        default=False,
    )

    args_parser.add_argument(
        '--commit-hash',
        type=str,
        default=None,
        required=True
    )

    return args_parser.parse_args()
