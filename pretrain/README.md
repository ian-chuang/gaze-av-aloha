python pretrain.py --type foveated_vit --device cuda:0
python pretrain.py --type low_res_vit --device cuda:1

tensorboard --logdir logs/miniimagenet --port 6006