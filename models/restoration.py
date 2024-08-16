import torch
import torch.nn as nn
import utils
import torchvision
import os
from tqdm import tqdm

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = self.diffusion.results_dir
        # image_folder = os.path.join(self.args.image_folder)
        # os.makedirs(image_folder, exist_ok=True)
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for i, (x, y) in enumerate(val_bar):
                img_num = y[0]
                # print(f"starting processing from image {y[0].strip('_')[0]}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond) 
                x_output = inverse_data_transform(x_output)
                utils.logging.save_image(x_output, os.path.join(image_folder, img_num))
                # print(img_num)
                val_bar.set_description("{} | {}".format(self.diffusion.args.name, img_num))

    def diffusive_restoration(self, x_cond):
        x = torch.randn(x_cond[:, :3, :, :].size(), device=self.diffusion.device) ## 
        x_output = self.diffusion.sample_image(x_cond, x)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list

class DiffusiveRestoration_Fusion:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration_Fusion, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume_ir) and os.path.isfile(args.resume_vi) :
            self.diffusion.load_ddm_ckpt(args.resume_ir, args.resume_vi, ema=True)
            self.diffusion.model_ir.eval()
            self.diffusion.model_vi.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='snow', r=None):
        # image_folder = os.path.join(self.args.image_folder, self.config.data.dataset)
        image_folder = os.path.join(self.args.image_folder)
        os.makedirs(image_folder, exist_ok=True)
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for i, (x, y) in enumerate(val_bar):
                img_num = y[0].split('_')[0]
                # print(f"starting processing from image {y[0].strip('_')[0]}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :6, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{img_num}.png"))
                # print(img_num)
                val_bar.set_description("{} | {}".format(validation, f"{img_num}.png"))

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond[:, :3, :, :].size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
