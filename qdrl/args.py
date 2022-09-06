import argparse


def get_args():
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--num-epochs',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--batch-size',
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
        '--dataset-dir',
        type=str,
        required=True,
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

    args_parser.add_argument(
        '--dataloader-workers',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--validate-recall',
        action='store_true',
        default=False,
    )

    args_parser.add_argument(
        '--triplet-loss-margin',
        type=float,
        default=1.0
    )

    return args_parser.parse_args()
