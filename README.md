# elliptic_to_evolvegcn_dataset_conversion_utility
The utility converts the Elliptic graph dataset into format needed by EvolveGCN following Steps 0,1,2,3,4 at https://github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md

The utility takes in the (3) Elliptic dataset .csv files:
* elliptic_txs_classes.csv
* elliptic_txs_edgelist.csv
* elliptic_txs_features.csv

and writes out the ( ) .csv files needed by EvolveGCN:

Prerequisites:
Pandas

To run:
python convert.py --source_folder ./infolder --dest_folder ./outfolder 

Docker available?
Yes

Enjoy!
