
from os import path
import os
import numpy as np
import cv2
from tensorboardX import SummaryWriter
from path import Path
import scipy.misc,json
from pebble import ProcessPool
from collections import Counter
from tqdm import tqdm
import torch.utils.data as data
import numpy as np
from imageio import imread
import random,sys
import matplotlib.pyplot as plt
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import tqdm
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#
def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

class Mono_Depth(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(Mono_Depth, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        #print("die")
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        return(disp1)


###################################################################################################################################################


def Pconv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def Pupconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class Pose(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(Pose, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = Pconv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = Pconv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = Pconv(conv_planes[1], conv_planes[2])
        self.conv4 = Pconv(conv_planes[2], conv_planes[3])
        self.conv5 = Pconv(conv_planes[3], conv_planes[4])
        self.conv6 = Pconv(conv_planes[4], conv_planes[5])
        self.conv7 = Pconv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = Pupconv(conv_planes[4],   upconv_planes[0])
            self.upconv4 = Pupconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = Pupconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = Pupconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = Pupconv(upconv_planes[3], upconv_planes[4])

            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        
        input = [target_image]
        input.extend(ref_imgs)

        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp:
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]

            exp_mask4 = nn.functional.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = nn.functional.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = nn.functional.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = nn.functional.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None
        return exp_mask1, pose
###########################################################################################################################################################
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


##############################################################################################################################################################
pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics,intrinsics_inv, padding_mode='zeros'):

    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')


    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode="euler")  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img

#######################################################################################################################################################
# In[7]:


model1 = Mono_Depth()
model1=model1.to(device)
model2=Pose()
model2=model2.to(device)
optim_params = [

      {'params': model1.parameters(), 'lr':0.0001},
      {'params': model2.parameters(), 'lr':0.0001}
]
optimizer = torch.optim.Adam(optim_params)
########################################################################################################################################################
#intrinsics matrix caliberation
zoom_y=256/375
zoom_x=512/1242
u=np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02,2.163791e-01, 0.000000e+00,0.000000e+00, 1.000000e+00, 2.745884e-03])
P_rect = u.reshape( 3, 4)
P_rect[0]*= zoom_x
P_rect[1]*= zoom_y
P_rect[:,:3]
intrinsic_matrix=P_rect[:,:3]



#################################################################################################################################################################
#dataloader
def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return(data)
#train data loader
import torch.utils.data as data
from imageio import imread
from path import Path 
import random
def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt'
        self.scenes=[]
        for folder in open(scene_list_path):
            #print(folder)
            if folder[-2]=="2":
                p0='/kitti/'
                self.scenes.append(self.root/folder[:-1])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.crawl_folders(sequence_length)
        #print(self.scenes)
        
        
    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = intrinsic_matrix
            imgs = sorted(scene.files('*.png'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        intrinsics = np.array(intrinsic_matrix)
        train_data={}
        try:
            tgt_img = load_as_float(sample['tgt'])
            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
            return np.array(tgt_img,dtype=np.float32), np.array(ref_imgs,dtype=np.float32),np.array(intrinsics,dtype=np.float32)
            
        except:
            tgt_img=np.zeros((256,512,3))
            ref_imgs=[np.zeros((256,512,3)),np.zeros((256,512,3))]
            return np.array(tgt_img,dtype=np.float32), np.array(ref_imgs,dtype=np.float32),np.array(intrinsics,dtype=np.float32)
            

    def __len__(self):
        return len(self.samples)



#dataloader
dump_root=Path('dataset/kitti/')
train_set=SequenceFolder(root=dump_root,
    transform=None,
    seed=1,
    train=True,
    sequence_length=3
    )




# In[10]:


train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True,
    num_workers=0, pin_memory=False,drop_last=True)
epoch_size = len(train_loader)

print(epoch_size)
###########################################################################################################################################################
def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


# loss function combination of both L1 and ssim
def reprojection_loss(tgt_img,warp_img):
    ssim_loss=(1-ssim(tgt_img,warp_img))/2
    loss=0.85*(ssim_loss) + 0.15* torch.mean(torch.abs(tgt_img-warp_img))
    #min_loss,_=torch.min(loss,dim=0,keepdim=True)
    return loss
def photometric_reconstruction_loss(warp1,warp2,tgt_img, ref_imgs1,ref_imgs2,depth):
    smooth_loss=get_smooth_loss(depth,tgt_img)#.to(device)
    loss1=reprojection_loss(tgt_img,warp1)#.to(device)
    loss2=reprojection_loss(tgt_img,warp2)#.to(device)
    net_loss=loss1+loss2+0.05*smooth_loss
    return net_loss 


    #####################################################################################################################################


epoch=500
for xxxt in tqdm.tqdm( range(epoch)):
    ij=0+xxxt
    loss_epoch=0 
    if xxxt%2==0:
        try:
            torch.save(model1.state_dict(),'monodepth19/'+str(xxxt)+'subres18m1.pt')
            torch.save(model2.state_dict(), 'posenet19/'+str(xxxt)+'subres18m2.pt')
        except:
            print("sorry no memory")     
        print("saving finished")
    for i, (tgt_img, ref_imgs, intrinsics) in enumerate(train_loader):
        torch.backends.cudnn.enabled =True
        ztgt_img=tgt_img.permute(0,3,1,2)        
        ztgt_img=ztgt_img.to(device)
        ztgt_img=ztgt_img/255
        ref_img1=ref_imgs[:,0,:,:,:].to(device)
        ref_img2=ref_imgs[:,1,:,:,:].to(device)
        ref_img1=ref_img1.permute(0,3,1,2)
        ref_img2=ref_img2.permute(0,3,1,2)
        ref_img1=ref_img1/255
        ref_img2=ref_img2/255
        intrinsics = intrinsics.to(device)
        intrinsics_inv=intrinsics.inverse()
        #print(intrinsics_inv)
        intrinsics_inv=intrinsics_inv.to(device)
        disparities = model1(ztgt_img)
        depth=1/disparities
        pose = model2(ztgt_img,[ref_img1,ref_img2])
        warp1=inverse_warp(ref_img1, depth[:,0], pose[:,0], intrinsics,intrinsics_inv,padding_mode="zeros").to(device)
        warp2=inverse_warp(ref_img2, depth[:,0], pose[:,1], intrinsics,intrinsics_inv,padding_mode="zeros").to(device)
        loss_1 =photometric_reconstruction_loss(warp1,warp2,ztgt_img,ref_img1,ref_img2,depth)
        loss=loss_1
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch=loss_epoch+loss      
        if i%100==0:
            #print("kssaving")
            try:
                torch.save(model1.state_dict(), 'monodepth19_subres18m1.pt')
                torch.save(model2.state_dict(), 'posenet19_subres18m2.pt')
                print("saved")
            except:
                print("sorry no memory")
    print(loss_epoch)






