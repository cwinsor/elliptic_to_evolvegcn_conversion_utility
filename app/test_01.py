import os
import pandas as pd
import argparse


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(
        description="PyTorch Detection Training",
        add_help=add_help)

    parser.add_argument("--elliptic_folder",
        default="/dataset",
        type=str, help="folder with elliptic dataset")

    parser.add_argument("--evolve_folder",
        default="/dataset/evolvegcn_format",
        type=str, help="folder with evolvegcn dataset")

    return parser


def main(args):

    print("################")
    print("compare the features...")

    elliptic_features = pd.read_csv(
        os.path.join(args.elliptic_folder, "elliptic_txs_features.csv"),
        header=None)
    evolve_features = pd.read_csv(
        os.path.join(args.evolve_folder, "elliptic_txs_features.csv"),
        header=None)

    print(elliptic_features.shape)
    print(evolve_features.shape)

    print("################")
    print("compare the classes...")

    elliptic_classes = pd.read_csv(
        os.path.join(args.elliptic_folder, "elliptic_txs_classes.csv"),
        header=0)
    evolve_classes = pd.read_csv(
        os.path.join(args.evolve_folder, "elliptic_txs_classes.csv"),
        header=0)

    print(elliptic_classes.shape)
    print(evolve_classes.shape)
    print(elliptic_classes.head(3))
    print(evolve_classes.head(3))

    print("################")
    print("compare the edgelist...")

    elliptic_edgelist = pd.read_csv(
        os.path.join(args.elliptic_folder, "elliptic_txs_edgelist.csv"),
        header=0)
    evolve_edgelist = pd.read_csv(
        os.path.join(args.evolve_folder, "elliptic_txs_edgelist_timed.csv"),
        header=0)

    print(elliptic_edgelist.shape)
    print(evolve_edgelist.shape)
    print(elliptic_edgelist.head(3))
    print(evolve_edgelist.head(3))

    print("################")
    print("check the nodetime...")

    evolve_nodetime = pd.read_csv(
        os.path.join(args.evolve_folder, "elliptic_txs_nodetime.csv"),
        header=0)
    print(evolve_nodetime.head(3))

    print("################")
    print("check the orig2contiguous...")

    evolve_txs_orig2contiguous = pd.read_csv(
        os.path.join(args.evolve_folder, "elliptic_txs_orig2contiguos.csv"),
        header=0)
    print(evolve_txs_orig2contiguous.head(3))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)