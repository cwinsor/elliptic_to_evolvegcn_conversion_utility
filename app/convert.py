import os
import pandas as pd
import argparse


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(
        description="PyTorch Detection Training",
        add_help=add_help)

    parser.add_argument("--source_folder",
        default="/dataset",
        type=str, help="source folder having elliptic dataset")

    parser.add_argument("--dest_folder",
        default="/dataset/evolvegcn_format",
        type=str, help="target folder for converted dataset")

    return parser


def main(args):

    for k, v in dict(vars(args)).items():
        print("{} {}".format(k, v))

    if not os.path.exists(args.dest_folder):
        os.makedirs(args.dest_folder)

    ################
    print("Step 1: Create a file named elliptic_txs_orig2contiguos.csv and modify elliptic_txs_features.csv")
    file_path = os.path.join(args.source_folder, "elliptic_txs_features.csv")
    elliptic_txs_features = pd.read_csv(
        file_path,
        header=None)
    # elliptic_txs_features = elliptic_txs_features.loc[0:20,:].copy()

    num_feature_colums = len(elliptic_txs_features.columns)
    expected = 167
    assert num_feature_colums == expected, \
        ("feature columns expected {} observed {}".format(expected, num_feature_colums))

    to_float = lambda x: x[0].astype(float)
    elliptic_txs_features['node_id_original'] = elliptic_txs_features[0]
    elliptic_txs_features['node_id_zerobased_int'] = elliptic_txs_features.index
    elliptic_txs_features['node_id_zerobased_float'] = elliptic_txs_features[['node_id_zerobased_int']].apply(to_float, axis=1)
    elliptic_txs_features['timestep_int'] = elliptic_txs_features[1]
    elliptic_txs_features['timestep_float'] = elliptic_txs_features[['timestep_int']].apply(to_float, axis=1)

    elliptic_txs_features[0] = elliptic_txs_features['node_id_zerobased_float']
    elliptic_txs_features[1] = elliptic_txs_features['timestep_float']

    elliptic_txs_orig2contiguos = elliptic_txs_features[["node_id_original", "node_id_zerobased_int"]]
    elliptic_txs_orig2contiguos.rename(columns={
        "node_id_original": "originalId",
        "node_id_zerobased_int": "contiguosId"},
        inplace=True)

    # write to file
    elliptic_txs_features.iloc[:, 0:num_feature_colums].to_csv(
        os.path.join(args.dest_folder, "elliptic_txs_features.csv"),
        header=False, index=False)

    elliptic_txs_orig2contiguos.to_csv(
        os.path.join(args.dest_folder, "elliptic_txs_orig2contiguos.csv"),
        header=True, index=False)

    # we will be doing a bunch of lookups (old EllipseID to new EllipseID)
    # so set index and make a lambda
    elliptic_txs_orig2contiguos.set_index(
        keys='originalId',
        inplace=True, verify_integrity=True)
    map_node_id = lambda x: elliptic_txs_orig2contiguos.loc[x[0]]['contiguosId']

    ################
    print("Step 2: Modify elliptic_txs_classes.csv")
    file_path = os.path.join(args.source_folder, "elliptic_txs_classes.csv")
    elliptic_txs_classes = pd.read_csv(
        file_path,
        header=0)
    # elliptic_txs_classes = elliptic_txs_classes.loc[0:10,:].copy()

    elliptic_txs_classes.rename(columns={
        "txId": "old_txId",
        "class": "old_class"},
        inplace=True)

    map_class = lambda x: -1.0 if x[0] == "unknown" else (1.0 if x[0] == "1" else (0.0 if x[0] == "2" else ("yuk")))
    elliptic_txs_classes['class'] = elliptic_txs_classes.loc[:, ['old_class']].apply(map_class, axis=1)
    elliptic_txs_classes['txId'] = elliptic_txs_classes[['old_txId']].apply(map_node_id, axis=1)

    # write to file
    elliptic_txs_classes.loc[:, ['txId', 'class']].to_csv(
        os.path.join(args.dest_folder, "elliptic_txs_classes.csv"),
        header=True, index=False)

    ################
    print("Step 3: Create a file named elliptic_txs_nodetime.csv")
    elliptic_txs_nodetime = elliptic_txs_features[['node_id_zerobased_int', 'timestep_int']]
    elliptic_txs_nodetime.rename(columns={
        "node_id_zerobased_int": "txId",
        "timestep_int": "timestep"},
        inplace=True)

    minus_one = lambda x: x[0] - 1
    elliptic_txs_nodetime['timestep'] = elliptic_txs_nodetime[["timestep"]].apply(minus_one, axis=1)

    # write to file
    elliptic_txs_nodetime.to_csv(
        os.path.join(args.dest_folder, "elliptic_txs_nodetime.csv"),
        header=True, index=False)

    ################
    print("Step 4: Modify elliptic_txs_edgelist.csv and rename it to elliptic_txs_edgelist_timed.csv")
    elliptic_txs_edgelist_timed = pd.read_csv(
        os.path.join(args.source_folder, "elliptic_txs_edgelist.csv"),
        header=0)
    # elliptic_txs_edgelist_timed = elliptic_txs_edgelist_timed.loc[0:10,:].copy()

    elliptic_txs_edgelist_timed.rename(columns={
        "txId1": "old_txId1",
        "txId2": "old_txId2"},
        inplace=True)

    elliptic_txs_edgelist_timed['txId1'] = elliptic_txs_edgelist_timed[['old_txId1']].apply(map_node_id, axis=1)
    elliptic_txs_edgelist_timed['txId2'] = elliptic_txs_edgelist_timed[['old_txId2']].apply(map_node_id, axis=1)

    # we will be looking up via the features table - set index for this purpose
    elliptic_txs_features.set_index(
        keys='node_id_zerobased_int',
        inplace=True, verify_integrity=True)

    # sanity check - nodes specified by edge should have the same timestep
    get_timestep_int = lambda x: elliptic_txs_features.loc[x[0]]["timestep_int"]
    elliptic_txs_edgelist_timed['timestep1_int'] = elliptic_txs_edgelist_timed[['txId1']].apply(get_timestep_int, axis=1)
    elliptic_txs_edgelist_timed['timestep2_int'] = elliptic_txs_edgelist_timed[['txId2']].apply(get_timestep_int, axis=1)
    elliptic_txs_edgelist_timed['ts_are_different'] = elliptic_txs_edgelist_timed['timestep1_int'] != elliptic_txs_edgelist_timed['timestep2_int']
    if elliptic_txs_edgelist_timed['ts_are_different'].any():
        df = elliptic_txs_edgelist_timed
        print("count total: {}".format(df.shape[0]))
        select = df['not_equal']
        print("count of mismatches = {}".format(df[select].shape[0]))
        print(df[select])
        assert df['not_equal'].any() is False, "yikes - fails check that adjacent nodes will have same timestep"

    # they want float timestep - whatever.
    get_timestep_float = lambda x: elliptic_txs_features.loc[x[0]]["timestep_float"]
    elliptic_txs_edgelist_timed['timestep'] = elliptic_txs_edgelist_timed[['txId1']].apply(get_timestep_float, axis=1)

    # write to file
    elliptic_txs_edgelist_timed[['txId1', 'txId2', 'timestep']].to_csv(
        os.path.join(args.dest_folder, "elliptic_txs_edgelist_timed.csv"),
        header=True, index=False)

    print('done')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
