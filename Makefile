# Makefile

# train

DATA_PATH='dataset/Reann_MPSC/jyryu/lmdb/train/mpsc.lmdb.train' 
OUTPUT_DIR='output/mpsc/train/250718_2128'
MODEL_PATH_PRE='checkpoint-9.pth'

# eval

DATA_PATH_EVAL='dataset/Reann_MPSC/jyryu/lmdb/test/mpsc.lmdb.test' # test dataset path
MODEL_PATH_FINE='output/mpsc/train/250718_1959/checkpoint-97.pth' # load checkpoint
OUTPUT_DIR_EVAL='trash' # save outputs


# Set the path to save checkpoints
# OUTPUT_DIR_PRE='output/pretrain_dig'
# path to imagenet-1k train set
# DATA_PATH_PRE='/path/to/pretrain_data/'


train:
	# batch_size can be adjusted according to the graphics card
	CUDA_VISIBLE_DEVICES=2,3,4,5 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 run_mae_pretraining_moco.py \
			--image_alone_path ${DATA_PATH} \
			--mask_ratio 0.7 \
			--batch_size 64 \
			--opt adamw \
			--output_dir ${OUTPUT_DIR} \
			--epochs 10 \
			--warmup_steps 100 \
			--max_len 25 \
			--num_view 2 \
			--moco_dim 256 \
			--moco_mlp_dim 4096 \
			--moco_m 0.99 \
			--moco_m_cos \
			--moco_t 0.2 \
			--num_windows 4 \
			--contrast_warmup_steps 0 \
			--contrast_start_epoch 0 \
			--loss_weight_pixel 1. \
			--loss_weight_contrast 0.1 \
			--only_mim_on_ori_img \
			--weight_decay 0.1 \
			--opt_betas 0.9 0.999 \
			--model pretrain_simmim_moco_ori_vit_small_patch4_32x128 \
			--patchnet_name no_patchtrans \
			--encoder_type vit \
			

run: 
# batch_size can be adjusted according to the graphics card
	CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 10040 run_class_finetuning.py \
		--model simmim_vit_small_patch4_32x128 \
		--data_path ${DATA_PATH} \
		--eval_data_path ${DATA_PATH} \
		--finetune ${MODEL_PATH_PRE} \
		--output_dir ${OUTPUT_DIR} \
		--batch_size 64 \
		--opt adamw \
		--opt_betas 0.9 0.999 \
		--weight_decay 0.05 \
		--data_set image_lmdb \
		--nb_classes 97 \
		--smoothing 0. \
		--max_len 25 \
		--epochs 140 \
		--warmup_epochs 1 \
		--drop 0.1 \
		--attn_drop_rate 0.1 \
		--drop_path 0.1 \
		--lr 1e-4 \
		--num_samples 1 \
		--fixed_encoder_layers 0 \
		--decoder_name attention \
		--decoder_type attention \
		--use_abi_aug \
		--num_view 2 \
		--disable_eval_during_finetuning \
	
	
eval:
	CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 10041 run_class_finetuning.py \
		--model simmim_vit_small_patch4_32x128 \
		--data_path ${DATA_PATH_EVAL} \
		--eval_data_path ${DATA_PATH_EVAL} \
		--output_dir ${OUTPUT_DIR_EVAL} \
		--batch_size 512 \
		--opt adamw \
		--opt_betas 0.9 0.999 \
		--weight_decay 0.05 \
		--data_set image_lmdb \
		--nb_classes 97 \
		--smoothing 0. \
		--max_len 25 \
		--resume ${MODEL_PATH_FINE} \
		--eval \
		--epochs 20 \
		--warmup_epochs 2 \
		--drop 0.1 \
		--attn_drop_rate 0.1 \
		--dist_eval \
		--num_samples 1000000 \
		--fixed_encoder_layers 0 \
		--decoder_name attention \
		--decoder_type attention \
		--beam_width 0 \
		--num_workers 0 \
	
		


.PHONY: run eval
