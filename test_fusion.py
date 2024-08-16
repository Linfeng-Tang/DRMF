import argparse
import os
import yaml
import torch
import numpy as np
import datasets
from models import DenoisingDiffusion_Fusion as DenoisingDiffusion_Fusion

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default='Fusion.yml',help="Path to the config file")
    parser.add_argument("--phase", type=str, default='fusion', help="val(generation)")
    parser.add_argument("--fusion_type", type=str, default='IVIF', help="fusion type options: ['IVIF', 'MIF']")
    parser.add_argument('--name', type=str, default='IVIF_Norm', help='subfolder name to save outputs') 
    parser.add_argument("--data_dir", type=str, default='data/IVIF_Norm/val', help="root data path")   
    parser.add_argument('--resume_ir', default='experiments/IVIF_norm/IR_MSRS.pth', type=str,help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--resume_vi', default='experiments/IVIF_norm/VI_LOL.pth', type=str,help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--resume_weight', default='experiments/Fusion/Fusion_DPCM.pth', type=str,help='Path for the diffusion model checkpoint to load for evaluation') 
    parser.add_argument('--save_folder', default='./Results', type=str,help='Folder for saving the fusion results')
    parser.add_argument("--sampling_timesteps", type=int, default=5, help="Number of implicit sampling steps")
    parser.add_argument('--seed', default=61, type=int, metavar='N',help='Seed for initializing training (default: 61)')
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
    val_loader = DATASET.get_test_loaders(parse_patches=False, phase='fusion', data_dir=args.data_dir)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion_Fusion(args, config)
    diffusion.sample_validation_eval(val_loader)


if __name__ == '__main__':
    main()