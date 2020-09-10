# In the training code we have given comments for various lines to facilitate the understanding. Please read the 'train.py' first.
opt = {'switch':20000}
opt.update({'lr':1e-4})

opt['dir_root']='/home/mohit/Music/attention_low_light/'
opt['exp_name'] = 'AVG-sir-amplifier'

opt['gpu'] = "1"
opt['epochs'] = 1000000
opt['batch_size'] = 1
opt['Shuffle'] = False
opt['Pin_memory'] = True
opt['workers'] = 1
opt['patch'] = 512

opt['fig_freq'] = 2000
opt['save_freq'] = [2,200000,400000,450000]
opt['text_prnt_freq']=2000

opt['fig_size'] = 5



import random
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torchvision.transforms.functional as Ft

import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
import skimage
from skimage.transform import resize
import time
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_root_mse as NRMSE
from torch.autograd import Variable
from math import exp
import math
import rawpy
import glob
import imageio


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
##os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=opt['gpu']



a = (np.logspace(0,8,64, endpoint=True, base=2.0)-1)/255

print(a)

img = np.random.random_sample((3, 2))
print(img)

values, edges  = np.histogram(img, bins=a, range=(0,1), normed=None, weights=None, density=None)
values = values/(img.shape[0]*img.shape[1])
print(values)



class get_data(Dataset):
    """Loads the Data."""
    
    def __init__(self,opt):
        self.train_files = glob.glob('/media/data/mohit/chen_dark_cvpr_18_dataset/Sony/short/1*_00_0.1s.ARW')
        

        self.gt_files = []
        for x in self.train_files:
            self.gt_files =self.gt_files+ glob.glob('/media/data/mohit/chen_dark_cvpr_18_dataset/Sony/long/*'+x[-17:-12]+'*.ARW')
        
        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        
        self.opt = opt
        
        self.a = (np.logspace(0,8,64, endpoint=True, base=2.0)-1)/255
        
    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        
       
        raw = rawpy.imread(self.gt_files[idx])
        
        img_gt = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).copy()
        img_gtt=np.float32(img_gt/65535.0)

        raw.close()
        
        raw = rawpy.imread(self.train_files[idx])
        img = raw.raw_image_visible.astype(np.float32).copy()
        raw.close()
        
        img_loww = (np.maximum(img - 512,0)/ (16383 - 512))
        H,W = img_loww.shape
        
        ##############################################################################
        
        low=[]

        gt=[]
        
        for_amplifier=[]

        for gener in range(1):
            
            if random.randint(0, 100)>50:
                flip_flag = True
            else:
                flip_flag = False

            if random.randint(0, 100)<20:
                v_flag = True
            else:
                v_flag = False

        #     print(H)

            i = 0#random.randint(0, (H-self.opt['patch']-2)//2)*2
            j = 0#random.randint(0,(W-self.opt['patch']-2)//2)*2

            img_low = img_loww#[i:i+self.opt['patch'],j:j+self.opt['patch']]
            img_gt = img_gtt#[i:i+self.opt['patch'],j:j+self.opt['patch'],:]
            
            
            
            if False:
                img_gt = np.flip(img_gt, 0).copy()
                img_low = np.flip(img_low, 0).copy()

            if False:
                img_gt = np.flip(img_gt, 1).copy()
                img_low = np.flip(img_low, 1).copy()
                
            values, edges  = np.histogram(img_low, bins=self.a, range=(0,1), normed=None, weights=None, density=None)
            values = values/(img_low.shape[0]*img_low.shape[1])

#            img_low_r = img_low[0:H:2,0:W:2]
#            img_low_g1 = img_low[0:H:2,1:W:2]
#            img_low_g2 = img_low[1:H:2,0:W:2]
#            img_low_b = img_low[1:H:2,1:W:2]
            
            for_amplifier.append(torch.from_numpy(values).float())
            
# We used the following code to do Pack operation in the training code. We used FOR loop for understanding sake. In the test code however we do it the vectorised way later in the code.
#            img_gt_avg = np.zeros((H//8,W//8,int(64*3))).astype(np.float32)

#            r_avg = np.zeros((H//16,W//16,64)).astype(np.float32)
#            g1_avg = np.zeros((H//16,W//16,64)).astype(np.float32)
#            g2_avg = np.zeros((H//16,W//16,64)).astype(np.float32)
#            b_avg = np.zeros((H//16,W//16,64)).astype(np.float32)

#            count_gt=0
#            count_raw = 0
#            for ii in range(8):
#                for jj in range(8):

#                    img_gt_avg[:,:,count_gt:count_gt+3] = img_gt[ii:H:8,jj:W:8,:]
#                    count_gt=count_gt+3
#        #             print(count_gt)

##                    r_avg[:,:,count_raw] = img_low_r[ii:H//2:8,jj:W//2:8]
##                    g1_avg[:,:,count_raw] = img_low_g1[ii:H//2:8,jj:W//2:8]
##                    g2_avg[:,:,count_raw] = img_low_g2[ii:H//2:8,jj:W//2:8]
##                    b_avg[:,:,count_raw] = img_low_b[ii:H//2:8,jj:W//2:8]
#                    count_raw=count_raw+1
#        #             print('{},{},{}'.format(count_raw,ii,jj))

            
            gt.append(torch.from_numpy((np.transpose(img_gt, [2, 0, 1]))).float())
            low.append(torch.from_numpy(img_low).float().unsqueeze(0))
#            r_low.append(torch.from_numpy((np.transpose(r_avg, [2, 0, 1]))).float())
#            g1_low.append(torch.from_numpy((np.transpose(g1_avg, [2, 0, 1]))).float())
#            g2_low.append(torch.from_numpy((np.transpose(g2_avg, [2, 0, 1]))).float())
#            b_low.append(torch.from_numpy((np.transpose(b_avg, [2, 0, 1]))).float())
            
        
        return gt, low, for_amplifier
    

obj_train = get_data(opt)
dataloader_train = DataLoader(obj_train, batch_size=opt['batch_size'], shuffle=opt['Shuffle'], num_workers=opt['workers'], pin_memory=opt['Pin_memory'])

#for i,img in enumerate(dataloader_train):
#    gt = img[0]
#    r_low = img[1]
#    g1_low = img[2]
#    g2_low = img[3]
#    b_low = img[4]
#    for_amplifier = img[5]
#    m = nn.PixelShuffle(8)

#    imageio.imwrite('GT.jpg',resize(gt[0].reshape(-1,8,3,512//8,512//8).permute(2,3,0,4,1).reshape(1,3,512,512)[0].cpu().numpy().transpose(1,2,0)*255,(512,512)).astype(np.uint8))
#    imageio.imwrite('ip.jpg',resize(m(g1_low[0])[0].cpu().numpy().transpose(1,2,0)*20*255,(512,512)).astype(np.uint8))
#    
#    
#    break
#    
#print(r_low[0].size())
#print(gt[0].size())
#print(for_amplifier[0].size())




class ResBlock(nn.Module):
    
    def __init__(self,in_c):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_c,in_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_c,in_c, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        
        return out
    
class make_dense(nn.Module):
    
    def __init__(self, nChannels=64, growthRate=32, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels=64, nDenselayer=6, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out
    

    
class amplifier(nn.Module):
    
    def __init__(self):
        super(amplifier, self).__init__()
        
        self.relu = nn.Threshold(threshold=0, value=0.0001, inplace=True)
        
        self.conv_post = nn.Sequential(
            torch.nn.Linear(63, 128),
            torch.nn.Linear(128, 1)
        )
        
        
    def forward(self, x):
        

        gamma = self.relu(self.conv_post(x))

        return gamma
    


    
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.amplifier = amplifier()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.RDBr = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.RDBg1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.RDBg2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.RDBb = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.before_identity = nn.Conv2d(in_channels=int(4*64), out_channels=64, kernel_size=1, stride=1, bias=False)
        self.after_rdb = nn.Conv2d(in_channels=int(3*64), out_channels=64, kernel_size=1, stride=1, bias=False)
        
        self.RDB1 = RDB(nChannels=64, nDenselayer=6, growthRate=32)
        self.RDB2 = RDB(nChannels=64, nDenselayer=6, growthRate=32)
        self.RDB3 = RDB(nChannels=64, nDenselayer=6, growthRate=32)
        
        self.final = nn.Sequential(
            nn.PixelShuffle(2),
            RDB(nChannels=16, nDenselayer=6, growthRate=32),
            nn.Conv2d(in_channels=16, out_channels=int(64*3), kernel_size=3, stride=1, padding=1, bias=True),
#             self.relu
        ) 
        
        
        
    def forward(self,low,for_amplifier):
        

        gamma = self.amplifier(for_amplifier)
        
        b,c,h,w = low.size()
        r=2
        
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        
        low = low.contiguous().view(b, c, out_h, r, out_w, r)
        low = low.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)
        
        r_low = low[0,0,:,:].unsqueeze(0).unsqueeze(0)
        g1_low = low[0,1,:,:].unsqueeze(0).unsqueeze(0)
        g2_low = low[0,2,:,:].unsqueeze(0).unsqueeze(0)
        b_low = low[0,3,:,:].unsqueeze(0).unsqueeze(0)
        
        b,c,h,w = r_low.size()
        
        r=8 # from here begins the Pack 8x operation
        
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        
        r_low = r_low.contiguous().view(b, c, out_h, r, out_w, r)
        r_low = r_low.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)
        #print(r_low.size())
        g1_low = g1_low.contiguous().view(b, c, out_h, r, out_w, r)
        g1_low = g1_low.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)
        
        g2_low = g2_low.contiguous().view(b, c, out_h, r, out_w, r)
        g2_low = g2_low.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)
        
        b_low = b_low.contiguous().view(b, c, out_h, r, out_w, r)
        b_low = b_low.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)
        
        r_low = self.relu(self.RDBr(r_low*gamma))
        g1_low = self.relu(self.RDBg1(g1_low*gamma))
        g2_low = self.relu(self.RDBg2(g2_low*gamma))
        b_low = self.relu(self.RDBb(b_low*gamma))
        
        alll=self.before_identity(torch.cat((r_low,g1_low,g2_low,b_low),dim=1))
        
        identity = alll
        
        rdb1 = self.RDB1(alll)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        
        alll = self.after_rdb(torch.cat((rdb1,rdb2,rdb3),dim=1))+identity
        
        alll = self.final(alll)
        alll = self.relu(alll.reshape(-1,8,3,356,532).permute(2,3,0,4,1).reshape(1,3,2848,4256)) # THIS IS THE ONE LINE UNPACK operation.
        
        
        return alll,gamma
    
    
    
class common_functions():
    
    def __init__(self, opt):
        
        self.opt = opt
        self.count = 0
        self.relu = nn.ReLU(inplace=True)
                        
        self.device = torch.device("cpu")
        
        model = Net()
        print('Trainable parameters : {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        
        self.model = model.to(self.device)
#        print(self.model)
        print(next(self.model.parameters()).is_cuda)  
        
        checkpoint = torch.load(self.opt['dir_root']+'weights/'+self.opt['exp_name']+'_{}'.format(400000))
        self.model.load_state_dict(checkpoint['model'])
        
        self.psnr = 0
        self.ssim = 0


  
    
    def optimize_parameters(self,low,gt,for_amplifier):
        
#         num = random.randint(0, 191)
#         self.num = (num//3)*3 
        
        low=low.to(self.device)
#        g1_low=g1_low.to(self.device)
#        g2_low=g2_low.to(self.device)
#        b_low=b_low.to(self.device)
        gt=gt.to(self.device)
        for_amplifier=for_amplifier.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            beg = time.time()
            pred,gamma = self.model(low,for_amplifier)
            end = time.time()
            #print('rlow {}'.format(r_low.size()))
            #print('pred {}'.format(pred.size()))
        
        
        
        
        plot_out_GT = gt#.reshape(-1,8,3,356,532).permute(2,3,0,4,1).reshape(1,3,2848,4256)
        plot_out_pred = pred
#        torch.zeros(1,3,2848,4256, dtype=torch.float).to(self.device)
#        counttt=0
#        for ii in range(8):
#                for jj in range(8):

#                    plot_out_GT[:,:,ii:2848:8,jj:4256:8] = gt[:,counttt:counttt+3,:,:]
#                    plot_out_pred[:,:,ii:2848:8,jj:4256:8] = pred[:,counttt:counttt+3,:,:]
#                    
#                    counttt=counttt+3
#        
        
#        plot_out_pred = self.relu(plot_out_pred)
        
         
        self.count +=1
        
            
            
        if True:
            
            plot_out_pred = (np.clip(plot_out_pred[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
            plot_out_GT = (np.clip(plot_out_GT[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
            
            psnrr = PSNR(plot_out_GT,plot_out_pred)
            ssimm = SSIM(plot_out_GT,plot_out_pred,multichannel=True)

            
            
            print("PSNR: {0:.3f}, SSIM: {1:.3f}, RMSE:{2:.3f}".format(psnrr,ssimm ,NRMSE(plot_out_GT,plot_out_pred)))
            
            self.psnr += psnrr
            self.ssim += ssimm
            
            print('Mean PSNR: {}'.format(self.psnr/self.count))
            print('Mean SSIM: {}'.format(self.ssim/self.count))


#            imageio.imwrite('/media/mohit/data/mohit/chen_dark_cvpr_18_dataset/Sony/results/sir_amplifier/{}_IMG_PRED.jpg'.format(self.count), plot_out_pred)
 #           imageio.imwrite('/media/mohit/data/mohit/chen_dark_cvpr_18_dataset/Sony/results/sir_amplifier/{}_IMG_GT.jpg'.format(self.count), plot_out_GT)
                        
            
            
            print(gamma)
            
        
    
    
    
    
    
gan_model = common_functions(opt)


for iteration, img in enumerate(dataloader_train):
    gt = img[0]
    low = img[1]
#    g1_low = img[2]
#    g2_low = img[3]
#    b_low = img[4]
    for_amplifier = img[2]
        
    for faster in range(1):
        gan_model.optimize_parameters(low[faster],gt[faster],for_amplifier[faster])
                      
                
    
            


