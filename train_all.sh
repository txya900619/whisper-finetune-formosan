#!/bin/bash

WANDB_PROJECT="whisper-finetune-indigenous" torchrun --nproc_per_node=4 train/fine-tune_on_hf_dataset_all_config_have_translation.py \
--model_name openai/whisper-large-v2 \
--language id \
--train_datasets ithuan/fb_ilrdf_dict_asr_clean ithuan/formosan_db_clean ithuan/ithuan_formosan_train ithuan/klokah_asr_train_clean \
--eval_datasets ithuan/formosan_org_eval_clean ithuan/klokah_asr_eval_clean \
--sampling_rate 16000 \
--num_proc 8 \
--train_strategy epoch \
--learning_rate 5e-4 \
--train_batchsize 2 \
--eval_batchsize 8 \
--num_epoch 5 \
--resume_from_ckpt None \
--output_dir outputs/schedule_free_r_5e-4_all_large_v2 \
--gradient_accumulation_steps 64


