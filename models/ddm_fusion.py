import os
import time
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import make_grid, save_image
import torch.backends.cudnn as cudnn
import utils
from utils import logger, metrics, losses
from models.unet import DiffusionUNet, WeightUNet
from datetime import datetime
import logging
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    if len(X.shape) > 3:
        B, C, _, _ = X.shape
        max_values, _ = torch.max(X.view(B, C, -1), dim=-1, keepdim=True)
        min_values, _ = torch.min(X.view(B, C, -1), dim=-1, keepdim=True)
        max_values = max_values.view(B, C, 1, 1)
        min_values = min_values.view(B, C, 1, 1)
        normalized_x = (X - min_values) / (max_values - min_values)
    else:        
        C, _, _ = X.shape
        max_values, _ = torch.max(X.view(C, -1), dim=-1, keepdim=True)
        min_values, _ = torch.min(X.view(C, -1), dim=-1, keepdim=True)
        max_values = max_values.view(C, 1, 1)
        min_values = min_values.view(C, 1, 1)
        normalized_x = (X - min_values) / (max_values - min_values)
    # normalized_x = torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
    
    return normalized_x


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

class DenoisingDiffusion_Fusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model_vi = DiffusionUNet(config)
        # self.model_vi = torch.nn.DataParallel(self.model_vi)
        self.model_vi.to(self.device)
        self.model_ir = DiffusionUNet(config)
        # self.model_ir = torch.nn.DataParallel(self.model_ir) 
        self.model_ir.to(self.device)
        
        
        self.model_weight = WeightUNet(config)
        # self.model_weight = torch.nn.DataParallel(self.model_weight)
        self.model_weight.to(self.device)
        
        self.loss_fusion_func = losses.Fusion_mask_loss(self.device)
        self.loss_weight_func = losses.Smooth_loss(self.device)
        self.loss_edge_func = losses.Edge_loss(self.device)
        self.loss_rec_func = nn.L1Loss().to(self.device)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model_weight)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model_weight.parameters()) ## Fine-tuning the weight distribution network Updating only the weights of the weight distribution network
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.setup(args, config)
        
    def load_ddm_ckpt(self, load_path_ir, load_path_vi, load_path_weight=None, ema=False):
        def remove_module_prefix(state_dict):
            # 如果有 'module.' 前缀，去掉它
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        checkpoint_ir = utils.logging.load_checkpoint(load_path_ir, None)
        checkpoint_vi = utils.logging.load_checkpoint(load_path_vi, None) 
        
        # 移除 'module.' 前缀后加载模型
        self.model_ir.load_state_dict(remove_module_prefix(checkpoint_ir['state_dict']), strict=True)
        self.model_vi.load_state_dict(remove_module_prefix(checkpoint_vi['state_dict']), strict=True)
        
        if load_path_weight is not None:
            checkpoint_weight = utils.logging.load_checkpoint(load_path_weight, None)
            self.model_weight.load_state_dict(remove_module_prefix(checkpoint_weight['state_dict']), strict=True)
            self.start_epoch = checkpoint_weight['epoch']
            self.step = checkpoint_weight['step']
            print("=> loaded checkpoint '{}' for diffusion prior combination model".format(load_path_weight))
        else:
            self.start_epoch = 0
            self.step = 0
        
        if ema:
            self.ema_helper.ema(self.model_weight)
        
        print("=> loaded checkpoint '{}' for infrared/CT restoration model".format(load_path_ir))
        print("=> loaded checkpoint '{}' for visible/MRI restoration model".format(load_path_vi))
    
        
    def setup(self, args, config):
        # self.experiments_root = os.path.join("experiments", '{}_{}'.format(args.name, datetime.now().strftime('%y%m%d_%H%M%S')))
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
                self.results_dir = os.path.join(self.args.save_folder, '{}'.format(args.name))
            else:
                self.results_dir = os.path.join(self.experiments_root, config.path.results)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            
            logger.setup_logger(None, self.log_dir,
                            'train', level=logging.INFO, screen=True)
            logger.setup_logger('val', self.log_dir, 'val', level=logging.INFO)
            self.logger = logging.getLogger('base')                                
            self.logger_val = logging.getLogger('val')  # validation logger
            
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_fusion_loaders()
        # print(self.args.resume)
        self.load_ddm_ckpt(self.args.resume_ir, self.args.resume_vi, self.args.resume_weight)
        self.model_ir.eval()      
        self.model_vi.eval()          
        best_psnr = 0        
        current_step = self.step
        current_epoch = self.start_epoch 
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            current_epoch += 1
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                current_step += 1
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model_weight.train()
                self.step += 1
                x = x.to(self.device)
                x_cond_A = x[:, :3, :, :].to(self.device)
                x_cond_B = x[:, 3:6, :, :].to(self.device)
                x_gt_A = x[:, 6:9, ::]
                x_gt_B = x[:, 9:12, ::]
                x_mask = x[:, 12:13, ::]
                x_mask = x_mask 
                x_edge = x[:, 13:14, ::]
                x_cond_A = data_transform(x_cond_A)
                x_cond_B = data_transform(x_cond_B)
                x_gt_A = data_transform(x_gt_A)
                x_gt_B = data_transform(x_gt_B) 
                
                shape = x_gt_A.shape
                x = torch.randn(size=shape, device=self.device)
                x_conds = [x_cond_A, x_cond_B]  
                results = self.sample_image_train(x_conds, x) 
                xt_preds = results['xs']
                x0_preds = results['x0']
                weights = results['weight']
                # print(torch.max(torch.cat(weights)), torch.max(x_mask))
                edges = results['edge']
                times = results['time'] 
                x0_irs = results['x0_ir']
                x0_vis = results['x0_vi']
                # times = [1 - time / self.num_timesteps for time in times]
                fusion_losses = self.loss_fusion_func(x0_preds[1:], x_gt_A, x_gt_B, times[1:], x_mask) ## 计算loss的时候输入的是GT图像啊
                fusion_loss = fusion_losses['loss_fusion']
                smooth_loss = self.loss_weight_func(weights[1:], x_mask)
                rec_loss_ir = 0
                for i, (x0_ir, weight_step) in enumerate(zip(x0_irs, times)):
                    if not i==0:
                        rec_loss_ir += (1 - weight_step) * self.loss_rec_func(x0_ir, x_gt_A)
                rec_loss_vi = 0
                for i, (x0_vi, weight_step) in enumerate(zip(x0_vis, times)):
                    if not i==0:
                        rec_loss_vi += (1 - weight_step) * self.loss_rec_func(x0_vi, x_gt_B)
                rec_loss = rec_loss_ir * 5 + rec_loss_vi * 1
                        
                loss = 10 * fusion_loss + 1 * smooth_loss + 0 * rec_loss                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model_weight)
                data_start = time.time()
                if self.step % 50 == 0:
                    ## write log and tensorboard
                    message = '<epoch:{:3d}, iter:{:8,d}, loss: {:.4f}, fusion loss: {:.4f}, smooth loss: {:.4f}, Rec infrared loss: {:.4f}, Rec Visible loss: {:.4f}, data time: {:.4f} > '.format(current_epoch, current_step, loss.item(), fusion_loss.item(), smooth_loss.item(), rec_loss_ir.item(), rec_loss_vi.item(), data_time / (i+1))
                    self.logger.info(message)                    
                    self.tb_logger.add_scalar('loss', loss.item(), current_step)
                    self.tb_logger.add_scalar('fusion_loss', fusion_loss.item(), current_step)
                    self.tb_logger.add_scalar('smooth_loss', smooth_loss.item(), current_step)
                    
                    tb_img = [inverse_data_transform(x_cond_A[0, ::].detach().float().cpu()), inverse_data_transform(x_cond_B[0, ::].detach().float().cpu()), inverse_data_transform(x_gt_A[0, ::].detach().float().cpu()), inverse_data_transform(x_gt_B[0, ::].detach().float().cpu()), torch.cat((x_mask, x_mask, x_mask), dim=1)[0, ::].detach().float().cpu()]
                    tb_img = make_grid(tb_img, nrow=len(tb_img), padding=2)
                    tb_xt = [inverse_data_transform(x[0, ::].detach().float().cpu()) for x in xt_preds]
                    tb_x0 = [inverse_data_transform(x[0, ::].detach().float().cpu()) for x in x0_preds]
                    tb_weight = [x[0, ::].detach().float().cpu() for x in weights]
                    tb_xt = make_grid(tb_xt, nrow=len(tb_xt), padding=2)
                    tb_x0 = make_grid(tb_x0, nrow=len(tb_x0), padding=2)
                    tb_weight = make_grid(tb_weight, nrow=len(tb_weight), padding=2)
                    
                    self.tb_logger.add_image('images', tb_img, current_step)
                    self.tb_logger.add_image('xt', tb_xt, current_step)
                    self.tb_logger.add_image('x0', tb_x0, current_step)
                    self.tb_logger.add_image('weight', tb_weight, current_step)
                    
                if self.step % self.config.training.validation_freq == 0:
                    self.model_weight.eval()
                    avg_psnr, avg_ssim = self.sample_validation_eval(val_loader)                    
                    self.tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    self.tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        ckpt_save_path = os.path.join(self.checkpoint_dir,  'Fusion_best')
                        self.logger.info('Saving best_psnr models and training states in {}.'.format(ckpt_save_path) )
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'state_dict': self.model_weight.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=ckpt_save_path) 
                         
                if self.step % self.config.training.snapshot_freq == 0:
                    ckpt_save_path = os.path.join(self.checkpoint_dir, 'Fusion_final')
                    self.logger.info('Saving final models and training states in {}.'.format(ckpt_save_path))
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model_weight.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=ckpt_save_path)
                    
                if self.step % self.config.training.ckpt_freq == 0:
                    ckpt_save_path = os.path.join(self.checkpoint_dir, 'Fusion_{:06d}'.format(self.step))
                    self.logger.info('Saving final models and training states in {}.'.format(ckpt_save_path))
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model_weight.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=ckpt_save_path)

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        models = [self.model_ir, self.model_vi]
        model_weight = self.model_weight
        results = utils.sampling.generalized_steps_multi_weight(x, x_cond, seq, models, model_weight=model_weight, b=self.betas, eta=0.0)
        if last:
            xs = results['xs'][-1]
        return xs, results
    
    def sample_image_mif(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        models = [self.model_ir, self.model_vi]
        model_weight = self.model_weight
        results = utils.sampling.generalized_steps_mif(x, x_cond, seq, models, model_weight=model_weight, b=self.betas, eta=0.0)
        if last:
            xs = results['xs'][-1]
        return xs, results
    
    def sample_image_train(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        models = [self.model_ir, self.model_vi]
        model_weight = self.model_weight
        results = utils.sampling.generalized_steps_multi_weight_train(x, x_cond, seq, models, model_weight=model_weight, b=self.betas, eta=0.0)
        return results
    
                
    def sample_validation(self, val_loader):  
        self.model_weight.eval()
        image_folder = os.path.join(self.results_dir, '{:06d}'.format(self.step))
        avg_psnr_val = 0.0
        avg_ssim_val = 0.0
        idx = 0
        # with torch.no_grad():
        if idx==0:
            self.logger.info(f"Processing a single batch of validation images at step: {self.step}")            
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
                x_cond_A = x[:, :3, :, :].to(self.device)
                x_cond_B = x[:, 3:6, :, :].to(self.device)
                x_gt_A = x[:, 6:9, ::]
                x_gt_B = x[:, 9:12, ::]
                x_cond_A = data_transform(x_cond_A)
                x_cond_B = data_transform(x_cond_B)
                
                shape = x_gt_A.shape
                x = torch.randn(size=shape, device=self.device)
                x_conds = [x_cond_A, x_cond_B]                
                pred_x, results = self.sample_image(x_conds, x)
                pred_x = inverse_data_transform(pred_x[:, :, h, w])

                for i in range(n):
                    avg_psnr += metrics.calculate_psnr(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt_A[i].permute(1, 2, 0).numpy() * 255.0, test_y_channel=True) + metrics.calculate_psnr(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt_B[i].permute(1, 2, 0).numpy() * 255.0, test_y_channel=True)
                    avg_ssim +=  metrics.calculate_ssim(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt_A[i].permute(1, 2, 0).numpy() * 255.0) + metrics.calculate_ssim(pred_x[i].permute(1, 2, 0).numpy() * 255.0, x_gt_B[i].permute(1, 2, 0).numpy() * 255.0)                    
                    utils.logging.save_image(pred_x[i], os.path.join(image_folder, y[i]))
                    val_bar.set_description("{} | {}".format(self.args.name, y[i]))
                avg_psnr_val += avg_psnr / n
                avg_ssim_val += avg_ssim / n
            avg_psnr_val = avg_psnr_val / idx
            avg_ssim_val = avg_ssim_val / idx
            self.logger.info("Average PSNR: {:04f}, Average SSIM: {:04f}".format(avg_psnr_val, avg_ssim_val))     
            return avg_psnr_val, avg_ssim_val 
    
    def sample_validation_eval(self, val_loader): 
        
        self.load_ddm_ckpt(self.args.resume_ir, self.args.resume_vi, self.args.resume_weight)
        self.model_ir.eval()      
        self.model_vi.eval()     
        self.model_weight.eval()
        image_folder = os.path.join(self.results_dir)        
        avg_psnr_val = 0.0
        avg_ssim_val = 0.0
        idx = 0
        # with torch.no_grad():
        if idx==0:  
            val_bar = tqdm(val_loader)
            for _, (x, y) in enumerate(val_bar): 
                idx += 1
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x                
                n = x.size(0)                 
                _, _, h, w = x.shape
                multiple = 16
                crop_height = int(multiple * np.ceil(h / multiple))
                crop_width = int(multiple * np.ceil(w / multiple))
                x = F.pad(x, (0, crop_width - w, 0, crop_height - h), mode='reflect')
                x_cond_A = x[:, :3, :, :].to(self.device)
                x_cond_B = x[:, 3:6, :, :].to(self.device)
                x_cond_A = data_transform(x_cond_A)
                x_cond_B = data_transform(x_cond_B)
                
                shape = x_cond_A.shape
                x = torch.randn(size=shape, device=self.device)
                x_conds = [x_cond_A, x_cond_B]              
                if self.args.fusion_type == 'MIF':  
                    pred_x, results = self.sample_image_mif(x_conds, x)
                else:                    
                    pred_x, results = self.sample_image(x_conds, x)
                pred_x = inverse_data_transform(pred_x[:, :, :h, :w])
                for i in range(n):
                    utils.logging.save_image(pred_x[i], os.path.join(image_folder, y[i])) 
                    val_bar.set_description("{} | {}".format(self.args.name, y[i]))
                
            print(os.path.join(image_folder, y[i]))   
            return avg_psnr_val, avg_ssim_val
