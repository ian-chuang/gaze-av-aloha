## Pretrained Models

Pretrained checkpoints are available on the Hugging Face Hub:

* [`iantc104/mae_vitb_vit`](https://huggingface.co/iantc104/mae_vitb_vit)
* [`iantc104/mae_vitb_low_res_vit`](https://huggingface.co/iantc104/mae_vitb_low_res_vit)
* [`iantc104/mae_vitb_foveated_vit`](https://huggingface.co/iantc104/mae_vitb_foveated_vit)

---

## Load Pretrained Foveated ViT

```python
from gaze_av_aloha.policies.gaze_policy.tokenizer import FoveatedImageTokenizer
from gaze_av_aloha.policies.gaze_policy.vit import create_vit_b

tokenizer = FoveatedImageTokenizer()
vit = create_vit_b(tokenizer.get_num_tokens(), tokenizer.get_token_size())
vit = vit.from_pretrained("iantc104/mae_vitb_foveated_vit")
```

---

## Pretraining

To run MAE pretraining with multiple GPUs:

```bash
# Fine (standard) ViT
python parallel_pretrain.py --type vit

# Coarse (low-res) ViT
python parallel_pretrain.py --type low_res_vit

# Foveated ViT
python parallel_pretrain.py --type foveated_vit
```

---

## Monitor Training

Launch TensorBoard to visualize training progress:

```bash
tensorboard --logdir logs/miniimagenet --port 6006
```

---

## Push Checkpoint to Hugging Face Hub

```bash
python push_to_hub.py \
  --type foveated_vit \
  --checkpoint vit-b-mae_foveated_vit.pth \
  --repo_id <your_username>/mae_vitb_foveated_vit
```
