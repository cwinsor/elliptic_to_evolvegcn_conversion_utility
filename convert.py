r"""Preprocessing utility for Elliptic dataset for use with EvolveGCN.

This code implements preprocessing Step0, 1, 2, 3, 4 identified in:
https://github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md

The steps convert the Elliptic dataset, downloadable
from https://www.kaggle.com/ellipticco/elliptic-data-set
into a format that can be consumed by the EvolveGCN code, specifically,
the dataloader elliptic_temporal_dl.py in https://github.com/IBM/EvolveGCN. 

To run:
clone this repo
download the Elliptic dataset to folder A
identify a target folder B

run the utility as:
python convert.py

requires:
python3

a requirements.txt file is included.

"""

import datetime
import os
import time
from datetime import timedelta
import logging


import pandas as pd

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--source_folder", default="/folder_with_elliptic_dataset", type=str, help="source folder having elliptic dataset")
    parser.add_argument("--dest_folder", default="/folder_for_converted_dataset", type=str, help="target folder for converted dataset")

    return parser


def main(args):

    for k, v in dict(vars(args)).items():
        print("{} {}".format(k,v))

    # Step 1: Create a file named elliptic_txs_orig2contiguos.csv and modify elliptic_txs_features.csv.
    file_path = os.path.join(args.source_folder, "elliptic_txs_features.csv")
    elliptic_txs_features = pd.read_csv(file_path, header=None)
    elliptic_txs_features = elliptic_txs_features.loc[0:20,:].copy()

    num_feature_colums = len(elliptic_txs_features.columns)
    expected = 167
    assert num_feature_colums==expected, ("feature columns expected {} observed {}".format(expected, num_feature_colums))

    to_float = lambda arg: arg.astype(float)
    elliptic_txs_features['node_id_original'] = elliptic_txs_features[0]
    elliptic_txs_features['node_id_zerobased_int'] = elliptic_txs_features.index
    elliptic_txs_features['node_id_zerobased_float'] = elliptic_txs_features.loc[:,['node_id_zerobased_int']].apply(to_float, axis=1)
    elliptic_txs_features['timestep_int'] = elliptic_txs_features[1]
    elliptic_txs_features['timestep_float'] = elliptic_txs_features.loc[:,['timestep_int']].apply(to_float, axis=1)

    elliptic_txs_features[0] = elliptic_txs_features['node_id_zerobased_float']
    elliptic_txs_features[1] = elliptic_txs_features['timestep_float']

    elliptic_txs_orig2contiguos = elliptic_txs_features[["node_id_original","node_id_zerobased_int"]]
    elliptic_txs_orig2contiguos.rename(columns={
        "node_id_original" : "originalId",
        "node_id_zerobased_int" : "contiguosId"},
        inplace=True)

    # write to file
    elliptic_txs_features.iloc[:,0:num_feature_colums].to_csv(
        os.path.join(args.dest_folder, "elliptic_txs_features.csv"),
        header=False, index=False)

    elliptic_txs_orig2contiguos.to_csv(
        os.path.join(args.dest_folder, "elliptic_txs_orig2contiguos.csv"),
        header=True, index=False)


    # Step 2: Modify elliptic_txs_classes.csv
    file_path = os.path.join(args.source_folder, "elliptic_txs_classes.csv")
    elliptic_txs_classes = pd.read_csv(file_path, header=0)
    elliptic_txs_classes = elliptic_txs_classes.loc[0:10,:].copy()

    map_class = lambda x: -1.0 if x[0]=="unknown" else (1.0 if x[0]=="1" else (0.0 if x[0]=="2" else ("yuk")))
    elliptic_txs_classes['new_class'] = elliptic_txs_classes.loc[:,['class']].apply(map_class, axis=1)

    # set index in preparation for a bunch of lookups (old EllipseID to new EllipseID)
    elliptic_txs_orig2contiguos.set_index('originalId', inplace=True)
    # map_node_id = lambda row: str(type(row))
    # map_node_id = lambda row: str(type(elliptic_txs_orig2contiguos.loc[orig_id]))
    # map_node_id = lambda row: row.values[0]
    # map_node_id = lambda row: str(type(elliptic_txs_orig2contiguos.loc[row.values[0]]))
    # map_node_id = lambda row: str(elliptic_txs_orig2contiguos.loc[row.values[0]])
    map_node_id = lambda row: elliptic_txs_orig2contiguos.loc[row.values[0]]['contiguosId']

    elliptic_txs_classes['new_tx_id'] = elliptic_txs_classes.apply(map_node_id, axis=1)
    # elliptic_txs_classes['new_tx_id'] = elliptic_txs_classes.loc[:,['txId']].apply(map_node_id, axis=1)
    # elliptic_txs_classes['new_tx_id'] = elliptic_txs_classes.loc[:,['txId']].apply(map_node_id, axis=1)

    print(elliptic_txs_classes)
    
    print(elliptic_txs_classes)




    # print (elliptic_txs_orig2contiguos)
    # print(elliptic_txs_orig2contiguos.loc[232344069,'contiguosId'])

    # map_node_id = lambda x: str(type(elliptic_txs_orig2contiguos.loc[x]))

    # map_node_id = lambda x: str(type(x))
    # map_node_id = lambda x: str(type(elliptic_txs_orig2contiguos.loc[x]))
    # map_node_id = lambda x: elliptic_txs_orig2contiguos.loc('contiguosId')
    # map_node_id = lambda x: 456
    xx = elliptic_txs_classes.loc[2,['txId']]
    print(xx.values[0])




    elliptic_txs_classes['new_tx_id'] = elliptic_txs_classes.loc[:,['txId']].apply(map_node_id, axis=1)

#    elliptic_txs_features['timestep_float'] = elliptic_txs_features.loc[:,['timestep_int']].apply(to_float, axis=1)


    print(elliptic_txs_classes)


    # elliptic_txs_orig2contiguos.loc[x,:])




    elliptic_txs_features['node_id_original'] = elliptic_txs_features[0]




# elliptic_txs_classes.csv





if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
