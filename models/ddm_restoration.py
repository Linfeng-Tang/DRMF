import os
import time
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import make_grid
import torch.backends.cudnn as cudnn
import utils
from utils import logger, metrics
from models.unet import DiffusionUNet, DiffusionUNet1
from datetime import datetime
import logging
import torch.nn.functional as F
from tensorboardX import SummaryWriter

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt() 
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float()) 
    pred_x0 = (x - output * (1.0 - a).sqrt()) / a.sqrt()
    loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return loss, x, output, pred_x0


class DenoisingDiffusion_Restoration(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.setup(self.args, self.config)        
        s, n = self.get_network_description(self.model)        
        net_struc_str = '{}'.format(self.model.__class__.__name__)
        self.logger.info( 'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        

    def load_ddm_ckpt(self, load_path, ema=False):
        def remove_module_prefix(state_dict):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(remove_module_prefix(checkpoint['state_dict']), strict=True)
        self.optimizer.load_state_dict(remove_module_prefix(checkpoint['optimizer']))
        self.ema_helper.load_state_dict(remove_module_prefix(checkpoint['ema_helper']))
        if ema:
            self.ema_helper.ema(self.model)
        self.logger.info("=> loaded checkpoint '{}'.".format(load_path))
        
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def setup(self, args, config):
        if args.phase == 'train':
            self.experiments_root = os.path.join("experiments", '{}'.format(args.name))
            self.log_dir = os.path.join(self.experiments_root, config.path.log)
            self.tb_logger_dir = os.path.join(self.experiments_root, config.path.tb_logger)
            self.results_dir = os.path.join(self.experiments_root, config.path.results)
            self.checkpoint_dir = os.path.join(self.experiments_root, config.path.checkpoint)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.tb_logger_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            logger.setup_logger(None, self.log_dir,
                            'train', level=logging.INFO, screen=True)
            logger.setup_logger('val', self.log_dir, 'val', level=logging.INFO)
            self.logger = logging.getLogger('base')                                
            self.logger_val = logging.getLogger('val')  # validation logger
            self.tb_logger = SummaryWriter(log_dir=self.tb_logger_dir)
        else:
            self.experiments_root = os.path.join("experiments", '{}'.format(args.name))
            self.log_dir = os.path.join(self.experiments_root, config.path.log)
            if self.args.save_folder is not None:
                self.results_dir = os.path.join(self.args.save_folder, args.name)
            else:
                self.results_dir = os.path.join(self.experiments_root, config.path.results)
            # self.results_dir = '/data/timer/DRMF/data/LLVIP_enhanced/ir'
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            
            logger.setup_logger(None, self.log_dir,
                            'train', level=logging.INFO, screen=True)
            logger.setup_logger('val', self.log_dir, 'val', level=logging.INFO)
            self.logger = logging.getLogger('base')                                
            self.logger_val = logging.getLogger('val')  # validation logger
    
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)
        best_psnr = 0        
        current_step = self.step
        current_epoch = self.start_epoch 
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            current_epoch += 1
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                current_step += 1
                # print("input shape:", x.shape)
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                # print("input shape:", x.shape)
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, :3, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss, x_t, pred_noise, pred_x0 = noise_estimation_loss(self.model, x, t, e, b)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
                if self.step % 100 == 0:
                    ## write log and tensorboard
                    message = '<epoch:{:3d}, iter:{:8,d}, loss: {:.4f}, data time: {:.4f} > '.format(current_epoch, current_step, loss.item(), data_time / (i+1))
                    self.logger.info(message)
                    ## write tensorboard
                    self.tb_logger.add_scalar('loss', loss.item(), current_step)
                    tb_img = [inverse_data_transform(x[0, :3, :, :].detach().float().cpu()), inverse_data_transform(x_t[0, ::].detach().float().cpu()),
                            inverse_data_transform(e[0, ::].detach().float().cpu()), inverse_data_transform(pred_noise[0, ::].detach().float().cpu()),
                            inverse_data_transform(pred_x0[0, ::].detach().float().cpu()), inverse_data_transform(x[0, 3:, ::].detach().float().cpu())]
                    tb_img = make_grid(tb_img, nrow=6, padding=2)
                    self.tb_logger.add_image('images', tb_img, current_step)
                    
                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    avg_psnr, avg_ssim = self.sample_validation_val(val_loader, self.step)    
                    self.logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    self.tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    self.tb_logger.add_scalar('ssim', avg_ssim, current_step)

                # if self.step % self.config.training.snapshot_freq == 0:
                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        ckpt_save_path = os.path.join(self.checkpoint_dir, self.config.data.dataset + '_' + self.args.name, 'best')
                        self.logger.info('Saving best_psnr models and training states in {}.'.format(ckpt_save_path) )
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=ckpt_save_path)
                         
                if self.step % self.config.training.snapshot_freq == 0:
                    ckpt_save_path = os.path.join(self.checkpoint_dir, self.config.data.dataset + '_' + self.args.name, 'final')
                    self.logger.info('Saving final models and training states in {}.'.format(ckpt_save_path))
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=ckpt_save_path)
                    
                if self.step % self.config.training.ckpt_freq == 0:
                    ckpt_save_path = os.path.join(self.checkpoint_dir, self.config.data.dataset + '_' + self.args.name, '{:06d}'.format(self.step))
                    self.logger.info('Saving final models and training states in {}.'.format(ckpt_save_path))
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=ckpt_save_path)
                    
    def sample_validation_val(self, val_loader, step):        
        image_folder = os.path.join(self.results_dir, "{:06d}".format(step))
        os.makedirs(image_folder, exist_ok=True)        
        avg_psnr_val = 0.0
        avg_ssim_val = 0.0
        idx = 0
        with torch.no_grad():
            self.logger.info(f"Processing a single batch of validation images at step: {step}")
            for _, (x, y) in enumerate(val_loader):
                idx += 1
                avg_psnr = 0.0
                avg_ssim = 0.0
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                
                n = x.size(0)
                
                _, _, h, w = x.shape
                multiple = 16
                crop_height = int(multiple * np.ceil(h / multiple))
                crop_width = int(multiple * np.ceil(w / multiple))
                x = F.pad(x, (0, crop_width - w, 0, crop_height - h), mode='reflect')   
                x_cond = x[:, :3, :, :].to(self.device)
                x_gt = x[:, 3:, ::]
                x_cond = data_transform(x_cond)
                
                shape = x_gt.shape
                x = torch.randn(size=shape, device=self.device)
                pred_x = self.sample_image(x_cond, x) ## This is the predicted x_0
                
                pred_x = inverse_data_transform(pred_x[:, :, :h, :w])
                x_cond = inverse_data_transform(x_cond[:, :, :h, :w]) ## Convert to image then compute psnr and ssim
                x_gt = x_gt[:, :, :h, :w]

                for i in range(n):
                    avg_psnr += metrics.calculate_psnr(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt[i].permute(1, 2, 0).numpy() * 255.0, test_y_channel=True)
                    avg_ssim +=  metrics.calculate_ssim(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt[i].permute(1, 2, 0).numpy() * 255.0)
                    utils.logging.save_image(pred_x[i], os.path.join(image_folder, "{}_fake.png".format(y[i][:-4])))
                avg_psnr_val += avg_psnr / n
                avg_ssim_val += avg_ssim / n
            avg_psnr_val = avg_psnr_val / idx
            avg_ssim_val = avg_ssim_val / idx
            return avg_psnr_val, avg_ssim_val
        
    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
        
    def sample_validation(self, val_loader):          
        self.load_ddm_ckpt(self.args.resume)
        self.model.eval()
        avg_psnr_val = 0.0
        avg_ssim_val = 0.0
        idx = 0
        with torch.no_grad():           
            val_bar = tqdm(val_loader)
            for _, (x, y) in enumerate(val_bar): 
                idx += 1
                avg_psnr = 0.0
                avg_ssim = 0.0
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                
                n = x.size(0)
                _, _, h, w = x.shape
                multiple = 16
                crop_height = int(multiple * np.ceil(h / multiple))
                crop_width = int(multiple * np.ceil(w / multiple))
                x = F.pad(x, (0, crop_width - w, 0, crop_height - h), mode='reflect')     
                x_cond = x[:, :3, :, :].to(self.device)
                x_gt = x[:, 3:, ::]
                x_cond = data_transform(x_cond)
                
                shape = x_gt.shape
                x = torch.randn(size=shape, device=self.device)
                pred_x = self.sample_image(x_cond, x) ## This is the predicted x_0
                
                pred_x = inverse_data_transform(pred_x[:, :, :h, :w])
                x_cond = inverse_data_transform(x_cond[:, :, :h, :w]) ## Convert to image then compute psnr and ssim
                x_gt = x_gt[:, :, :h, :w]

                for i in range(n):
                    avg_psnr += metrics.calculate_psnr(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt[i].permute(1, 2, 0).numpy() * 255.0, test_y_channel=True)
                    avg_ssim +=  metrics.calculate_ssim(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt[i].permute(1, 2, 0).numpy() * 255.0)
                    utils.logging.save_image(pred_x[i], os.path.join(self.results_dir, y[i]))                 
                    val_bar.set_description("{} | {}".format(self.args.name, y[i]))
                avg_psnr_val += avg_psnr / n
                avg_ssim_val += avg_ssim / n
            avg_psnr_val = avg_psnr_val / idx
            avg_ssim_val = avg_ssim_val / idx
            self.logger.info("Average PSNR: {:04f}, Average SSIM: {:04f}".format(avg_psnr_val, avg_ssim_val))       