## IAGCN

## Dataset:
Download the raw GPS trajectory data by following URL:

**METR-LA**:  https://github.com/liyaguang/DCRNN

**PEMS-BAY**:  https://github.com/liyaguang/DCRNN

**BJ-AIR**: https://quotsoft.net/air/

## Running code:

You can run the following command to train the model at different datasets:

**METR-LA**: python train.py --gcn_bool --addaptadj  --randomadj --dataset 'METR-LA' --no_nm 150 --n_m 50 --expid 1 --inductive_adp True

**PEMS-BAY**: python train.py --gcn_bool --addaptadj  --randomadj --dataset 'PEMS-BAY' --no_nm 210 --n_m 70 --expid 1 --inductive_adp True

**PEMSD7-L**: python train.py --gcn_bool --addaptadj  --randomadj --dataset 'PEMSD7-L' --no_nm 140 --n_m 68 --expid 1 --inductive_adp True

**BJ-AIR**: python train.py --gcn_bool --addaptadj  --randomadj --dataset 'BJ-AIR' --no_nm 21 --n_m 7 --expid 1 --inductive_adp True
