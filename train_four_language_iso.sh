#!/bin/bash

WANDB_PROJECT="whisper-finetune-indigenous" torchrun --nproc_per_node=4 train/fine-tune_on_hf_dataset_language_iso_prompt.py \
--model_name openai/whisper-large-v3 \
--language id \
--train_datasets formospeech/formosan_gmm_cleaned formospeech/formosan_gmm_cleaned formospeech/formosan_gmm_cleaned formospeech/formosan_gmm_cleaned \
--train_dataset_configs ami sdq trv pwn \
--train_dataset_splits train train train train \
--train_dataset_text_columns text text text text \
--eval_datasets formospeech/klokah_crawled_eval formospeech/klokah_crawled_eval formospeech/klokah_crawled_eval formospeech/klokah_crawled_eval \
--eval_dataset_configs 阿美 賽德克 太魯閣 排灣 \
--eval_dataset_splits train train train train \
--eval_dataset_text_columns text text text text \
--sampling_rate 16000 \
--num_proc 16 \
--train_strategy epoch \
--learning_rate 7.5e-5 \
--warmup_ratio 0.1 \
--train_batchsize 100 \
--eval_batchsize 100 \
--num_epoch 4 \
--resume_from_ckpt None \
--output_dir outputs/schedule_free_7.5e-5_four_language_iso_prompt


