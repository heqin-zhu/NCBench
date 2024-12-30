# examples: train and test
nohup python3 -m NCfold.main --run_name .runs/dim256 --batch_size 32 -g 0 --phase train --dim 256 --lr 0.0005 --nfolds 1 --index_name data_index.yaml --epoch 400  --Lmin 600 --training_set RNAStrAlign --data_dir SS_data > log_train_dim256 2>&1 &
nohup python3 -m NCfold.main --run_name .runs/dim256 --batch_size 32 -g 0 --phase test --dim 256 --nfolds 1 --index_name data_index.yaml --test_epoch 396 --test_set PDB_test archiveII --data_dir SS_data > log_test_dim256 2>&1 &
