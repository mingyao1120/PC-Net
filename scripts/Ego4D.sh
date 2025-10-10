slot_num_iteration=4
seed=8
co_cl_loss=0.4
num_props=8
inter_lambda=0.146
co_qua_loss=1

CUDA_VISIBLE_DEVICES=1 python train.py \
    --inter_lambda $inter_lambda \
    --seed $seed \
    --config-path config/ego4d/main.json \
    --log_dir LOG_DIR \
    --tag TAG \
    --slot_num_iteration $slot_num_iteration \
    --co_cl_loss $co_cl_loss \
    --num_props $num_props \

# ====================== eval
# resume='checkpoints/SOTA/ego4d/8_2025-04-24_04-24-01/model-best.pt' # replace this checkpoint
# # eval on Test
# python train.py --config-path config/ego4d/main.json --log_dir LOG_DIR --eval --slot_num_iteration $slot_num_iteration --resume $resume
