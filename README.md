# This is official Pytorch implementation of [DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior](https://dl.acm.org/doi/pdf/10.1145/3664647.3681064) (ACM MM 2024).

```
@inproceedings{Tang2024DRMF,
    title={DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior},
    author={Tang, Linfeng and Deng, Yuxin and Yi, Xunpeng and Yan, Qinglong and Yuan, Yixuan and Ma, Jiayi},
    booktitle={Proceedings of the ACM International Conference on Multimedia},
    pages={8546--8555},
    year={2024}
}
```

## Over Framework

<p align="center">
    <img src="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Framework.png" alt="Framework" style="display:inline-block; margin-right:20px; margin-bottom:20px;" />
</p>
<p align="center">
    The overall framework of the proposed DRMF
</p>

## Virtual Environment

```python
conda create -n DRMF python=3.9
conda install -r requirement.txt
```

## Fast Testing

1. Downloading the [pre-trained model](https://pan.baidu.com/s/1UZv9z6Gl9HuIXp3BHLEinQ?pwd=DRFM) and placing them in **./experiments** .
2. Run the following script for fusion testing:

```python
## for practical infrared and visible image fusion
python test_fusion.py --config 'Fusion.yml' \
               --phase 'fusion' \
               --fusion_type 'IVIF' \
               --name 'IVIF_norm' \
               --data_dir 'data/IVIF_norm/val' \
               --resume_ir 'experiments/IVIF_norm/IR_MSRS.pth' \
               --resume_vi 'experiments/IVIF_norm/VI_LOL.pth' \
               --resume_weight 'experiments/Fusion/Fusion_DPCM.pth' \
               --save_folder './Results' \
               --sampling_timesteps 5

## for challenging infrared and visible image fusion
python test_fusion.py --config 'Fusion.yml' \
               --phase 'fusion' \
               --fusion_type 'IVIF' \
               --name 'IVIF_degraded' \
               --data_dir 'data/IVIF_degraded/val' \
               --resume_ir 'experiments/IVIF_degraded/IR_MSRS.pth' \
               --resume_vi 'experiments/IVIF_degraded/VI_LOL.pth' \
               --resume_weight 'experiments/Fusion/Fusion_DPCM.pth' \
               --save_folder './Results' \
               --sampling_timesteps 5


## for practical medical image fusion
python test_fusion.py --config 'Fusion.yml' \
               --phase 'fusion' \
               --fusion_type 'MIF' \
               --name 'MIF_norm' \
               --data_dir 'data/MIF_norm/val' \
               --resume_ir 'experiments/MIF_norm/CT.pth' \
               --resume_vi 'experiments/MIF_norm/MRI.pth' \
               --resume_weight 'experiments/Fusion/Fusion_DPCM.pth' \
               --save_folder './Results' \
               --sampling_timesteps 5

## for challenging medical image fusion
python test_fusion.py --config 'Fusion.yml' \
               --phase 'fusion' \
               --fusion_type 'MIF' \
               --name 'MIF_degraded' \
               --data_dir 'data/MIF_degraded/val' \
               --resume_ir 'experiments/MIF_degraded/CT.pth' \
               --resume_vi 'experiments/MIF_degraded/MRI.pth' \
               --resume_weight 'experiments/Fusion/Fusion_DPCM.pth' \
               --save_folder './Results' \
               --sampling_timesteps 5
```

## Training

### Training DRCDMs

1. `Construct pairs of degraded images and their corresponding high-quality version. (For example, '.\data\LOL\train\high' and '.\data\LOL\train\low' for low-light image enhancement)`
2. Edite **./configs/Restoration.yml** for setting hyper-parameters.
3. Run the following script for training the DRCDMs:

```python
## for training DRCDM for visible images on the LOL dataset
python train_restoration.py --config 'Restoration.yml' \
                            --phase 'train' \
                            --data_type 'LOL' \
                            --name 'Restoration_VI_LOL' \
                            --data_dir 'data/LOL' \
                            --sampling_timesteps 20

## for training DRCDM for infrared images on the MSRS dataset
python train_restoration.py --config 'Restoration.yml' \
                            --phase 'train' \
                            --data_type 'IR' \
                            --name 'Restoration_IR_MSRS' \
                            --data_dir 'data/IVIF_degraded' \
                            --sampling_timesteps 20


## for training DRCDM for CT images on the Harvard dataset
python train_restoration.py --config 'Restoration.yml' \
                            --phase 'train' \
                            --data_type 'CT' \
                            --name 'Restoration_CT' \
                            --data_dir 'data/MIF_degraded' \
                            --sampling_timesteps 20


## for training DRCDM for MRI images on the Harvard dataset
python train_restoration.py --config 'Restoration.yml' \
                            --phase 'train' \
                            --data_type 'MRI' \
                            --name 'Restoration_MRI' \
                            --data_dir 'data/MIF_degraded' \
                            --sampling_timesteps 20
```

## Training DPCM

1. Run the following script to generate high quality reference images if high quality GTs are missing, such as normal visible images in the MSRS dataset:

```python
python test_restoration.py --config 'Restoration.yml' \
                           --phase 'test' \
                           --data_type 'VI' \
                           --name 'Restoration_VI_MSRS' \
                           --data_dir 'data/IVIF_degraded/train' \
                           --resume 'experiments/IVIF_degraded/VI_LOL.pth' \
                           --save_folder './Restoration' \
                           --sampling_timesteps 20 \
```

## Training DPCM

1. Run the following script to generate high quality reference images if high quality GTs are missing, such as normal visible images in the MSRS dataset:

```python
python test_restoration.py --config 'Restoration.yml' \
                           --phase 'test' \
                           --data_type 'VI' \
                           --name 'Restoration_VI_MSRS' \
                           --data_dir 'data/IVIF_degraded/train' \
                           --resume 'experiments/IVIF_degraded/VI_LOL.pth' \
                           --save_folder './Restoration' \
                           --sampling_timesteps 20
```

2. Construct pairs of degraded images and their corresponding high-quality version.
3. Editing **./configs/Fusion.yml** for setting hyper-parameters.
4. Run the following script for training the DPCM:

```python
## for training DPCM for information aggragation on the MSRS dataset
python your_script.py --config 'Fusion.yml' \
                      --phase 'train' \
                      --fusion_type 'IVIF' \
                      --name 'Fusion_DPCM' \
                      --data_dir 'data/IVIF_degraded' \
                      --resume_ir 'experiments/IVIF_degraded/IR_MSRS.pth' \
                      --resume_vi 'experiments/IVIF_degraded/VI_LOL.pth' \
                      --sampling_timesteps 5
```

## Motivation

<p align="center">
    <img src="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Example.png" alt="Demo"  style="display:inline-block; margin-right:20px; margin-bottom:20px;" />
</p>
<p align="center">
    Fusion schematic in challenging scenarios for MMIF tasks. DDFM and DIVFusion are Diffusion-based and illumination-robust image fusion method, respectively. DiffLL and CAT are SOTA image restoration approeches
</p>

## Experiments

### Qualitative fusion results

<p align="center">
    <img src="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/IVIF.png" alt="IVIF" style="display:inline-block; margin-right:20px; margin-bottom:20px;" />
</p>
<p align="center">
    Visual comparison of DRMF with state-of-the-art approaches on practical and challenging fusion scenarios for IVIF
</p>

<p align="center">
    <img src="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/MIF.png" alt="MIF" style="display:inline-block; margin-right:20px; margin-bottom:20px;" />
</p>
<p align="center">
    Visual comparison of our DRMF with state-of-the-art approaches on normal and challenging scenarios for MIF
</p>

### Quantitative fusion results

<p align="center">
    <img src="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Quantitative.png" alt="Quantitative" style="display:inline-block; margin-right:20px; margin-bottom:20px;" />
</p>
<p align="center">
    Quantitative comparison of DRMF with state-of-the-art methods on IVIF and MIF tasks
</p>

### Pre-enhancement for other fusion methods

<p align="center">
    <img src="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Pre-Enhancement.png" alt="Quantitative" style="display:inline-block; margin-right:20px; margin-bottom:20px;" />
</p>
<p align="center">
    Pre-enhancement for existing approaches
</p>

### Object Detection

<p align="center">
    <img src="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Detection.png" alt="Quantitative" style="display:inline-block; margin-right:20px; margin-bottom:20px;" />
</p>
<p align="center">
    Visual comparison of object detection on the LLVIP dataset
</p>

## Citions
If this work is helpful to you, please cite it as：

```
@inproceedings{Tang2024DRMF,
    title={DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior},
    author={Tang, Linfeng and Deng, Yuxin and Yi, Xunpeng and Yan, Qinglong and Yuan, Yixuan and Ma, Jiayi},
    booktitle={Proceedings of the ACM International Conference on Multimedia},
    pages={8546--8555},
    year={2024}
}
```

## Acknowledgements
This code is built on [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion).
