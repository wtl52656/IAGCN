## IAGCN

## Dataset:
Download the raw GPS trajectory data by following URL:

**METR-LA**:  https://github.com/liyaguang/DCRNN

**PEMS-BAY**:  https://github.com/liyaguang/DCRNN

**BJ-AIR**: https://quotsoft.net/air/

## Running code:

You can run the following command to train the model at different datasets:

**METR-LA**: python train.py --dataset 'METR-LA' --no_nu 150 --n_u 50 --inductive_adp True --pred_loss 0.05

**PEMS-BAY**: python train.py --dataset 'PEMS-BAY' --no_nu 210 --n_u 70 --expid 1 --inductive_adp True --pred_loss 0.05

**PEMSD7-L**: python train.py --dataset 'PEMSD7-L' --no_nu 150 --n_u 57 --inductive_adp True --pred_loss 0.05

**BJ-AIR**: python train.py --dataset 'BJ-AIR' --no_nu 21 --n_u 10 --inductive_adp True --pred_loss 0.05
