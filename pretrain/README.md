

CUDA_VISIBLE_DEVICES=0 python parallel_pretrain.py --type foveated_vit --port 12355 --add_noise
CUDA_VISIBLE_DEVICES=1 python parallel_pretrain.py --type low_res_vit  --port 12356
CUDA_VISIBLE_DEVICES=2 python parallel_pretrain.py --type foveated_vit --port 12357 --model_name vit-b-mae_no-noise
python parallel_pretrain.py --type vit 

tensorboard --logdir logs/miniimagenet --port 6006