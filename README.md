# This is official Pytorch implementation of [DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior](https://openreview.net/forum?id=BwXrlBweab) (ACM MM 2024).

```

@inproceedings{Tang2024DRMF,

    title={DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior},

    author={Tang, Linfeng and Deng, Yuxin and Yi, Xunpeng and Yan, Qinglong and Yuan, Yixuan and Ma, Jiayi},

    booktitle=ACMMM,

    year={2024}

}

```

## Over Framework

<div>

    [imgsrc=&#34;https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Framework.png&#34;alt=&#34;Framework&#34;style=&#34;display:inline-block;margin-right:20px;margin-bottom:20px;&#34;](imgsrc=%22https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Framework.png%22alt=%22Framework%22style=%22display:inline-block;margin-right:20px;margin-bottom:20px;%22)

</div>

<palign="center">

    `<em>`The overall framework of the proposed DRMF.`</em>`

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

1. Construct pairs of degraded images and their corresponding high-quality version. (For example, '.\data\LOL\train\high' and '.\data\LOL\train\low' for low-light image enhancement)
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

'''


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

                           --sampling_timesteps 20 \
```


3. Construct pairs of degraded images and their corresponding high-quality version.
4. Editing **./configs/Fusion.yml** for setting hyper-parameters.
5. Run the following script for training the DPCM:

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

<divalign="center">

    <imgsrc="https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Example.png"alt="Demo"width="800"  style="display:inline-block;margin-right:20px;margin-bottom:20px;">

</div>

## Experiments

### Qualitative fusion results

<divalign="center">

    [imgsrc=&#34;https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/IVIF.png&#34;alt=&#34;MSRS&#34;style=&#34;display:inline-block;margin-right:20px;margin-bottom:20px;&#34;](imgsrc=%22https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/IVIF.png%22alt=%22MSRS%22style=%22display:inline-block;margin-right:20px;margin-bottom:20px;%22)

</div>

<palign="center">

    `<em>`<spanstyle="font-size: 50px;">Visual comparison of DRMF with state-of-the-art approaches on practical and challenging fusion scenarios for IVIF`</em>`

</p>

<divalign="center">

    [imgsrc=&#34;https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/MIF.png&#34;alt=&#34;M3FD&#34;style=&#34;display:inline-block;margin-right:20px;margin-bottom:20px;&#34;](imgsrc=%22https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/MIF.png%22alt=%22M3FD%22style=%22display:inline-block;margin-right:20px;margin-bottom:20px;%22)

</div>

<palign="center">

    `<em>`<spanstyle="font-size: 50px;">Visual comparison of our DRMF with state-of-the-art approaches on normal and challenging scenarios for MIF.`</em>`

</p>

### Quantitative fusion results

<divalign="center">

    [imgsrc=&#34;https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Quantitative.png&#34;alt=&#34;MSRS&#34;style=&#34;display:inline-block;margin-right:20px;margin-bottom:20px;&#34;](imgsrc=%22https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Quantitative.png%22alt=%22MSRS%22style=%22display:inline-block;margin-right:20px;margin-bottom:20px;%22)

</div>

<palign="center">

    `<em>`<spanstyle="font-size: 50px;">Quantitative comparison of DRMF with state-of-the-art methods on IVIF and MIF tasks.`</em>`

</p>

### Pre-enhancement for other fusion methods

<divalign="center">

    [imgsrc=&#34;https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Pre-Enhancement.png&#34;alt=&#34;MSRS&#34;style=&#34;display:inline-block;margin-right:20px;margin-bottom:20px;&#34;](imgsrc=%22https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Pre-Enhancement.png%22alt=%22MSRS%22style=%22display:inline-block;margin-right:20px;margin-bottom:20px;%22)

</div>

<palign="center">

    `<em>`<spanstyle="font-size: 50px;">Pre-enhancement for existing approaches.`</em>`

</p>

### Object Detection

<divalign="center">

    [imgsrc=&#34;https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Detection.png&#34;alt=&#34;MSRS&#34;style=&#34;display:inline-block;margin-right:20px;margin-bottom:20px;&#34;](imgsrc=%22https://github.com/Linfeng-Tang/DRMF/blob/main/Figures/Detection.png%22alt=%22MSRS%22style=%22display:inline-block;margin-right:20px;margin-bottom:20px;%22)

</div>

<palign="center">

    `<em>`<spanstyle="font-size: 50px;">Visual comparison of object detection on the LLVIP dataset.`</em>`

</p>

## If this work is helpful to you, please cite it as：

```

@inproceedings{Tang2024DRMF,

    title={DRMF: Degradation-Robust Multi-Modal Image Fusion via Composable Diffusion Prior},

    author={Tang, Linfeng and Deng, Yuxin and Yi, Xunpeng and Yan, Qinglong and Yuan, Yixuan and Ma, Jiayi},

    booktitle=ACMMM,

    year={2024}

}

```
