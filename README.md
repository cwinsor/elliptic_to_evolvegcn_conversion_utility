# elliptic_to_evolvegcn_dataset_conversion_utility
The utility converts the Elliptic graph dataset into format needed by EvolveGCN following Steps 0,1,2,3,4 at https://github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md

Elliptic is a company specializing in identifying fradulent crypto transactions. For research purposes they have published the Elliptic dataset - sets of Bitcoin transactions in the form of a graph.
EvolveGCN is a machine learning-based graph analysis tool by researchers at MIT and IBM. Although it is designed to analyze the Elliptic data it requires preprocessing of that data - identified in the link above. The utility here performs the preprocessing.

The utility takes in the (3) Elliptic dataset .csv files:
* elliptic_txs_classes.csv
* elliptic_txs_edgelist.csv
* elliptic_txs_features.csv

and writes out the ( ) .csv files expected by EvolveGCN:

Prerequisites:
Docker

To run:
Make a folder. Download the Elliptic dataset into it and unzip. The dataset is at https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
cd into the folder
Run the docker image giving the current folder as path to dataset:
  docker run -it -v $PWD:/dataset ellipse_to_evolvegcn
Within the docker container run the conversion utility:
  python convert.py --source_folder /dataset --dest_folder /dataset/evolvegcn_format
Exit the docker container
  exit
From the host find the converted dataset
  ls ./evolvegcn_format


