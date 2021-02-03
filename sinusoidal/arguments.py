import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Robust MAML experiments")

    parser.add_argument(
        "--detector",
        type=str,
        default="bayes",
        choices=["bayes", "minimax", "neyman-pearson"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory where model and logs are saved",
    )
    parser.add_argument(
        "--skew_task_distribution",
        action="store_true",
        default=False,
        help="If skewed, set(T_train) & set(T_eval) = {}",
    )

    parser.add_argument(
        "--n_iter", type=int, default=50000, help="number of meta-iterations"
    )

    parser.add_argument("--tasks_per_metaupdate", type=int, default=25)

    parser.add_argument(
        "--k_meta_train",
        type=int,
        default=10,
        help="data points in task training set (during meta training, inner loop)",
    )
    parser.add_argument(
        "--k_meta_test",
        type=int,
        default=10,
        help="data points in task test set (during meta training, outer loop)",
    )
    parser.add_argument(
        "--k_shot_eval",
        type=int,
        default=10,
        help="data points in task training set (during evaluation)",
    )

    parser.add_argument(
        "--lr_inner",
        type=float,
        default=0.1,
        help="inner-loop learning rate (task-specific)",
    )
    parser.add_argument(
        "--lr_meta", type=float, default=0.001, help="outer-loop learning rate"
    )

    parser.add_argument(
        "--num_inner_updates",
        type=int,
        default=1,
        help="number of inner-loop updates (during training)",
    )

    parser.add_argument("--num_hidden_layers", type=int, nargs="+", default=[40, 40])

    parser.add_argument(
        "--first_order",
        action="store_true",
        default=False,
        help="run first-order version",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
