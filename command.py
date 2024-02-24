import os

os.system("python train.py --dataset 'METR-LA' --no_nm 150 --n_m 50 --expid 1 --inductive_adp True")
os.system("python train.py --dataset 'PEMS-BAY' --no_nm 210 --n_m 70 --expid 1 --inductive_adp True")
os.system("python train.py --dataset 'PEMSD7-L' --no_nm 140 --n_m 68 --expid 1 --inductive_adp True")
os.system("python train.py --dataset 'BJ-AIR' --no_nm 21 --n_m 7 --expid 1 --inductive_adp True")
