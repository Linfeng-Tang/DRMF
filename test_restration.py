import argparse
import os
import yaml
import torch
import numpy as np
import datasets
from models import DenoisingDiffusion_Restoration

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default='Restoration.yml', help="Path to the config file")
    parser.add_argument("--phase", type=str, default='test', help="val(generation)")
    parser.add_argument("--data_type", type=str, default='LOL', help="data type options: ['LOL', 'IR', 'VI', 'CT', 'MRI', 'CT_norm', MRI_norm']")    
    parser.add_argument("--data_dir", type=str, default='data/LOL/val', help="root data path")   
    parser.add_argument('--name', type=str, default='Restoration_VI_LOL', help='folder name to save outputs')
    parser.add_argument('--resume', default='experiments/IVIF_degraded/VI_LOL.pth', type=str, help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--save_folder', default='./Restoration', type=str, help='Folder for saving the fusion results')
    parser.add_argument("--sampling_timesteps", type=int, default=20, help="Number of implicit sampling steps")
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="1")
    parser.add_argument('--seed', default=66, type=int, metavar='N', help='Seed for initializing training (default: 61)')
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
    device = torch.device("cuda:{}".format(args.gpu_ids)) if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config, args)
    # phase='test', data_type='lol', data_dir=None
    val_loader = DATASET.get_test_loaders(parse_patches=False, phase='test', data_dir=args.data_dir)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion_Restoration(args, config)
    diffusion.sample_validation(val_loader)


if __name__ == '__main__':
    main()
