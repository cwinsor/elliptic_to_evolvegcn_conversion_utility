# elliptic_to_evolvegcn_dataset_conversion_utility
The utility converts the Elliptic graph dataset into format needed by EvolveGCN following Steps 0,1,2,3,4 at https://github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md

Elliptic is a company specializing in identifying fradulent crypto transactions. For research purposes they have published the Elliptic dataset - a subset of Bitcoin transactions in the form of a graph.
EvolveGCN is a machine learning-based graph analysis tool by researchers at MIT and IBM designed to analyze the Elliptic data. It requires preprocessing of the Elliptic data.  The utility here performs that preprocessing.

The utility takes the 3 Elliptic dataset .csv files:
* elliptic_txs_classes.csv
* elliptic_txs_edgelist.csv
* elliptic_txs_features.csv

and writes into a separate folder the 5 .csv files needed by EvolveGCN:
* elliptic_txs_classes.csv
* elliptic_txs_edgelist_timed.csv
* elliptic_txs_features.csv
* elliptic_txs_nodetime.csv
* elliptic_txs_orig2contiguos.csv

Prerequisites:
Docker

To run:
Make a folder with the 3 Elliptic dataset .csv files. The dataset is available at https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
cd into the folder
Run the docker image with the following command:
  docker run -it -v $PWD:/dataset ellipse_to_evolvegcn
Within the docker container run the conversion utility:
  python convert.py --source_folder /dataset --dest_folder /dataset/evolvegcn_format
The EvolveGCN files will be put in a subfolder ./evolvegcn_format
Exit the docker container
Back on the host the EvolveGCN dataset will be a sub-folder evolvegcn_format
  ls ./evolvegcn_format


