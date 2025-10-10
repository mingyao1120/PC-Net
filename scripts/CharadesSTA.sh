slot_num_iteration=4
seed=8
co_cl_loss=0.5
num_props=8
inter_lambda=0.146
co_qua_loss=1

# ========================= train
# CUDA_LAUNCH_BLOCKING=0 CUDA_VISIBLE_DEVICES=0 python train.py \
#     --inter_lambda $inter_lambda \
#     --seed $seed \
#     --config-path config/charades/main_train_test.json \
#     --log_dir LOG_DIR \
#     --tag TAG \
#     --slot_num_iteration $slot_num_iteration \
#     --co_cl_loss $co_cl_loss \
#     --num_props $num_props \
#     --co_qua_loss $co_qua_loss \


# ====================== eval
resume='checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt' # replace this checkpoint
# eval on Test-Trivial
python train.py --config-path config/charades/main_train_test.json --log_dir LOG_DIR --eval --slot_num_iteration $slot_num_iteration --resume $resume

# eval on Novel_Composition
python train.py --config-path config/charades/main_novel_comp.json --log_dir LOG_DIR --eval --slot_num_iteration $slot_num_iteration --resume $resume

# eval on Novel_Word
python train.py --config-path config/charades/main_novel_word.json --log_dir LOG_DIR --eval --slot_num_iteration $slot_num_iteration --resume $resume