### Classification Experiments
```bash
GPU=0
DATAROOT=None
TASK=tcga_brca_subtype
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25 --pretrain_4k vit4k_xs_dino
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0 --pretrain_4k vit4k_xs_dino
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25 --pretrain_4k vit4k_xs_dino --freeze_4k
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0 --pretrain_4k vit4k_xs_dino --freeze_4k
TASK=tcga_kidney_subtype
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25 --pretrain_4k vit4k_xs_dino
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0 --pretrain_4k vit4k_xs_dino
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25 --pretrain_4k vit4k_xs_dino --freeze_4k
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0 --pretrain_4k vit4k_xs_dino --freeze_4k
TASK=tcga_lung_subtype
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25 --pretrain_4k vit4k_xs_dino
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0 --pretrain_4k vit4k_xs_dino
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 0.25 --pretrain_4k vit4k_xs_dino --freeze_4k
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_lgp --task $TASK --prop 1.0 --pretrain_4k vit4k_xs_dino --freeze_4k
```

### Survival Experiments
```bash
GPU=0
CUDA_VISIBLE_DEVICES=$GPU python main.py --which_splits 5foldcv --split_dir tcga_blca --mode pyramid --model_type hipt_lgp --pretrain_4k vit4k_xs_dino --freeze_4k
CUDA_VISIBLE_DEVICES=$GPU python main.py --which_splits 5foldcv --split_dir tcga_brca --mode pyramid --model_type hipt_lgp --pretrain_4k vit4k_xs_dino --freeze_4k
CUDA_VISIBLE_DEVICES=$GPU python main.py --which_splits 5foldcv --split_dir tcga_coadread --mode pyramid --model_type hipt_lgp --pretrain_4k vit4k_xs_dino --freeze_4k
CUDA_VISIBLE_DEVICES=$GPU python main.py --which_splits 5foldcv --split_dir tcga_lung --mode pyramid --model_type hipt_lgp --pretrain_4k vit4k_xs_dino --freeze_4k
CUDA_VISIBLE_DEVICES=$GPU python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode pyramid --model_type hipt_n --pretrain_4k vit4k_xs_dino --freeze_4k
```