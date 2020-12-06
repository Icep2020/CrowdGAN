# CrowdGAN
This is a Pytorch implementation of TPAMI 2021 paper "CrowdGAN: Identity-free Interactive Crowd Video Generation and Beyond".
Liangyu Chai, Yongtuo Liu, Wenxi Liu, Guoqiang Han, and Shengfeng He*

## Requirements
Pytorch 1.4.0+, Python 3.6+

## Training
Pretrain the Spatial Transfer Generator (STG)
```shell
python3.6 train.py --name STG --model STG --batchSize 4 --max_steps 160000 --gpu_ids 0,1,2,3
```
Pretrain the Point-aware Flow Predictor (PFP)
```shell
python3.6 train.py --name PFP --model PFP --batchSize 4 --max_steps 400000 --gpu_ids 0,1,2,3 --flownet_ckpt checkpoints/flownet.pth
```
Train the whole model
```shell
python3.6 train.py --name Final --model Final --batchSize 4 --max_steps 200000 --gpu_ids 0,1,2,3  --flownet_ckpt checkpoints/flownet.pth --mapG_ckpt checkpoints/STG/latest_net_netG.pth --flowG_ckpt checkpoints/PFP/latest_net_netG.pth
```

## Inference
```shell
python3.6 test.py --name test --dataroot . --gpu_ids 0  --flownet_ckpt checkpoints/flownet.pth --netG_ckpt checkpoints/Final/latest_net_netG.pth --mapG_ckpt checkpoints/Final/latest_net_mapG.pth --flowG_ckpt checkpoints/Final/latest_net_flowG.pth
```
## Contact

Please let me know if you encounter any problem. [icepoint1018@gmail.com](icepoint1018@gmail.com)
