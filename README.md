```markdown
# ANN-SNN Conversion with Modified Activation and BPTT Fine-tuning

This is the official code for the paper **"Enhancing ANN-SNN Conversion: Addressing Low Latency and Negative Thresholds with Modified Activation and BPTT Fine-tuning"**.

## Requirements

- Python 3.x
- PyTorch == 1.12.1
- tqdm == 4.63.0
- numpy == 1.21.5
- torchvision == 0.13.1
- spikingjelly == 0.0.0.0.14

Install requirements:
```bash
pip install torch==1.12.1 tqdm==4.63.0 numpy==1.21.5 torchvision==0.13.1 spikingjelly==0.0.0.0.14
```

## Pre-trained Weights

You can download pre-trained weights from:  
🔗 [Google Drive Link](https://drive.google.com/drive/folders/1fjCQVKppxuBdV_7D5agLge4grR_OO7m9?usp=drive_link)

After downloading, place the weights in the `model/` directory.

## Usage

### 1. Train ANN weights

```bash
python main.py \
  --dataset CIFAR100 \
  --net_arch vgg16 \
  --L 8 \
  --trainann_epochs 300 \
  --batchsize 128 \
  --presim_len 4 \
  --sim_len 32 \
  --direct_training
```

### 2. Convert to SNN and inference

```bash
python run_snn.py \
  --dataset CIFAR100 \
  --net_arch vgg16 \
  --sn_type gn \
  --load_model_name model/QCFS_CIFAR100_vgg16_L4
```

### 3. (Optional) SNN Fine-tuning

If the accuracy is not satisfactory, you can fine-tune the SNN using `main.py` with appropriate parameters.

