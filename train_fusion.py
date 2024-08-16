import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np
import datasets
from models import DenoisingDiffusion_Fusion as DenoisingDiffusion_Fusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default='Fusion.yml', help="Path to the config file")
    parser.add_argument("--phase", type=str, default='train', help="val(generation)")
    parser.add_argument("--fusion_type", type=str, default='IVIF', help="fusion type options: ['IVIF', 'MIF']")
    parser.add_argument('--name', type=str, default='Fusion_DPCM', help='folder name to save outputs')
    parser.add_argument("--data_dir", type=str, default='data/IVIF_degraded', help="root data path")   
    parser.add_argument('--resume_ir', default='experiments/IVIF_degraded/IR_MSRS.pth', type=str,help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--resume_vi', default='experiments/IVIF_degraded/VI_LOL.pth', type=str,help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--resume_weight', default=None, type=str, help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=5, help="Number of implicit sampling steps for validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N', help='Seed for initializing training (default: 61)')    
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="1")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    # setup device to run
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config, args)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion_Fusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
