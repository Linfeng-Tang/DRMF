import torch
import torch.nn as nn
import torch.nn.functional as F

def RGB2YCrCb(rgb_image):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    Convert YcrCb format to RGB format
    :param Y.
    :param Cb.
    :param Cr.
    :return.
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out


class Fusion_loss(nn.Module):
    def __init__(self, device):
        super(Fusion_loss, self).__init__()
        print('Using Fusion_loss() as loss function~')
        self.sobelconv = sobel_operation().to(device)
        self.loss_func = nn.L1Loss(reduction='mean').to(device)

    def forward(self, imgs_fusion, img_A, img_B, weights):
        loss_intensity = 0
        loss_color = 0
        loss_grad = 0
        loss_fusion = 0
        for (img_fusion, weight) in zip(imgs_fusion, weights):
            Y_fusion, Cb_fusion, Cr_fusion = RGB2YCrCb(img_fusion)
            Y_A, Cb_A, Cr_A = RGB2YCrCb(img_A)
            Y_B, Cb_B, Cr_B = RGB2YCrCb(img_B)
            Y_joint = torch.max(Y_A, Y_B)
            loss_intensity = weight * (20 * self.loss_func(Y_fusion, Y_joint) + 0 * self.loss_func(Y_fusion, Y_A) + 0 * self.loss_func(Y_fusion, Y_B))
            loss_color = 100 * weight * (self.loss_func(Cb_fusion, Cb_B) + self.loss_func(Cr_fusion, Cr_B))
            grad_A = self.sobelconv(Y_A)
            grad_B = self.sobelconv(Y_B)
            grad_fusion = self.sobelconv(Y_fusion)

            grad_joint = torch.max(grad_A, grad_B)
            loss_grad += weight * (self.loss_func(grad_fusion, grad_joint) + 2 * self.loss_func(grad_fusion, grad_A))
            loss_fusion += (loss_intensity + loss_color + 20 * loss_grad)
        loss = {
            'loss_intensity' : loss_intensity,
            'loss_color' : loss_color,
            'loss_grad' : loss_grad,
            'loss_fusion' : loss_fusion
        }
        return loss

class Fusion_mask_loss(nn.Module):
    def __init__(self, device):
        super(Fusion_mask_loss, self).__init__()
        print('Using Fusion_mask_loss() as loss function~')
        self.sobelconv = sobel_operation().to(device)
        self.loss_func = nn.L1Loss(reduction='mean').to(device)

    def forward(self, imgs_fusion, img_A, img_B, weights, mask=None):
        loss_intensity = 0
        loss_color = 0
        loss_grad = 0
        loss_fusion = 0
        for i, (img_fusion, weight) in enumerate(zip(imgs_fusion, weights)):
            if i == len(img_fusion) - 1: 
                Y_fusion, Cb_fusion, Cr_fusion = RGB2YCrCb(img_fusion)
                Y_A, Cb_A, Cr_A = RGB2YCrCb(img_A)
                Y_B, Cb_B, Cr_B = RGB2YCrCb(img_B)
                Y_joint = torch.max(Y_A, Y_B)
                if mask is not None: 
                    loss_intensity = weight * (10 * self.loss_func(Y_fusion, Y_joint) + 40 * self.loss_func(mask * Y_fusion, mask * Y_A) + 5 * self.loss_func(Y_fusion * (1 - mask), Y_B * (1 - mask)))
                    loss_color = 1 * weight * (self.loss_func(Cb_fusion * (1 - mask), Cb_B * (1 - mask)) + self.loss_func(Cr_fusion * (1 - mask), Cr_B * (1 - mask)))
                else:
                    loss_intensity = weight * (0 * self.loss_func(Y_fusion, Y_joint) + 5 * self.loss_func(Y_fusion, Y_A) + 1 * self.loss_func(Y_fusion, Y_B))
                    loss_color = 1 * weight * (self.loss_func(Cb_fusion, Cb_B) + self.loss_func(Cr_fusion, Cr_B))
                grad_A = self.sobelconv(Y_A)
                grad_B = self.sobelconv(Y_B)
                grad_fusion = self.sobelconv(Y_fusion)

                grad_joint = torch.max(grad_A, grad_B)
                loss_grad += 10 * weight * self.loss_func(grad_fusion, grad_joint) +  50 * self.loss_func(mask * grad_fusion, mask * grad_A)
                loss_fusion += (loss_intensity + loss_color + loss_grad)
        loss = {
            'loss_intensity' : loss_intensity,
            'loss_color' : loss_color,
            'loss_grad' : loss_grad,
            'loss_fusion' : loss_fusion
        }
        return loss
   
class Smooth_loss(nn.Module):
    def __init__(self, device):
        super(Smooth_loss, self).__init__()
        print('Using Smooth_loss() for smooth regularity~')
        self.sobelconv = sobel_operation().to(device)
        self.loss_func = nn.BCELoss().to(device)
        

    def forward(self, weights, mask=None):
        loss_smooth = 0
        for weight in weights: 
            weight_grad = self.sobelconv(weight)
            loss_smooth += torch.mean(torch.abs(weight_grad))
            if mask is not None:
                loss_smooth += 5 * self.loss_func(weight, mask)
        return loss_smooth

class Edge_loss(nn.Module):
    def __init__(self, device):
        super(Edge_loss, self).__init__()
        print('Using Edge_loss() for constrain the edeg map regularity~')
        self.loss_func = nn.BCELoss().to(device)
        

    def forward(self, edges, mask):
        loss_edge = 0
        for edge in edges: ## 
            loss_edge += 1 * self.loss_func(edge, mask)
        return loss_edge      

class Grad_loss(nn.Module):
    def __init__(self):
        super(Grad_loss, self).__init__()
        self.sobelconv = sobel_operation()

    def forward(self, fuse_img, image_vis, image_ir):
        vis_grad = self.sobelconv(image_vis)
        ir_grad = self.sobelconv(image_ir)
        fuse_grad = self.sobelconv(fuse_img)

        grad_joint = torch.max(vis_grad, ir_grad)
        loss_grad = F.l1_loss(fuse_grad, grad_joint)
        return loss_grad

class sobel_operation(nn.Module):
    def __init__(self):
        super(sobel_operation, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)
        # self.weightx = kernelx.cuda()
        # self.weighty = kernely.cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)