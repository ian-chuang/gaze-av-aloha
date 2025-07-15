

python pretrain.py --type foveated_vit --device cuda:0
python pretrain.py --type low_res_vit --device cuda:1
CUDA_VISIBLE_DEVICES=1,2,3 python pretrain.py --type vit --use_parallel --device cuda --num_workers 16

tensorboard --logdir logs/miniimagenet --port 6006