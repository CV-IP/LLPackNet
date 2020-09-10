opt = {'switch':20000}
opt.update({'lr':1e-4})

opt['dir_root']='/home/mohit/Music/attention_low_light/' # directory where files and weights are stored
opt['exp_name'] = 'AVG-sir-amplifier' # name of the experiment python file and weights

opt['gpu'] = "1"
opt['epochs'] = 1000000 # This trains the code for very long time. We stopped the execution after 400,000 iterations MANUALLY
opt['batch_size'] = 1
opt['Shuffle'] = True # Should we load training images in random order  
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



a = (np.logspace(0,8,64, endpoint=True, base=2.0)-1)/255 # setting bin edges for the histogram

print(a)

img = np.random.random_sample((3, 2))
print(img)

values, edges  = np.histogram(img, bins=a, range=(0,1), normed=None, weights=None, density=None)
values = values/(img.shape[0]*img.shape[1])
print(values) # just to test the histogram and no relevance witht the code



class get_data(Dataset):
    """Loads the Data."""
    
    def __init__(self,opt):
        self.train_files = glob.glob('/media/data/mohit/chen_dark_cvpr_18_dataset/Sony/short/0*_00_0.1s.ARW')
        self.train_files = self.train_files + glob.glob('/media/data/mohit/chen_dark_cvpr_18_dataset/Sony/short/2*_00_0.1s.ARW') # The network takes days to train. We recommend you load the full dataset onto RAM which greatly reduces the training time. So choose wisely the number of images that can be loaded into your RAM.

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
        
        r_low=[]
        g1_low=[]
        g2_low=[]
        b_low=[]

        gt=[]
        
        for_amplifier=[]

        for gener in range(4):
            
            if random.randint(0, 100)>50:
                flip_flag = True
            else:
                flip_flag = False

            if random.randint(0, 100)<20:
                v_flag = True
            else:
                v_flag = False

        #     print(H)

            i = random.randint(0, (H-self.opt['patch']-2)//2)*2
            j = random.randint(0,(W-self.opt['patch']-2)//2)*2

            img_low = img_loww[i:i+self.opt['patch'],j:j+self.opt['patch']]
            img_gt = img_gtt[i:i+self.opt['patch'],j:j+self.opt['patch'],:]
            
            if flip_flag:
                img_gt = np.flip(img_gt, 0).copy()
                img_low = np.flip(img_low, 0).copy()

            if v_flag:
                img_gt = np.flip(img_gt, 1).copy()
                img_low = np.flip(img_low, 1).copy()
                
            values, edges  = np.histogram(img_low, bins=self.a, range=(0,1), normed=None, weights=None, density=None)
            values = values/(img_low.shape[0]*img_low.shape[1])

            img_low_r = img_low[0:self.opt['patch']:2,0:self.opt['patch']:2]
            img_low_g1 = img_low[0:self.opt['patch']:2,1:self.opt['patch']:2]
            img_low_g2 = img_low[1:self.opt['patch']:2,0:self.opt['patch']:2]
            img_low_b = img_low[1:self.opt['patch']:2,1:self.opt['patch']:2]
            
            for_amplifier.append(torch.from_numpy(values).float())
            

            img_gt_avg = np.zeros((opt['patch']//8,opt['patch']//8,int(64*3))).astype(np.float32)

            r_avg = np.zeros((opt['patch']//16,opt['patch']//16,64)).astype(np.float32)
            g1_avg = np.zeros((opt['patch']//16,opt['patch']//16,64)).astype(np.float32)
            g2_avg = np.zeros((opt['patch']//16,opt['patch']//16,64)).astype(np.float32)
            b_avg = np.zeros((opt['patch']//16,opt['patch']//16,64)).astype(np.float32)

            count_gt=0 # We now begin the Pack 8x operation
            count_raw = 0
            for ii in range(8):
                for jj in range(8):

                    img_gt_avg[:,:,count_gt:count_gt+3] = img_gt[ii:opt['patch']:8,jj:opt['patch']:8,:]
                    count_gt=count_gt+3
        #             print(count_gt)

                    r_avg[:,:,count_raw] = img_low_r[ii:opt['patch']//2:8,jj:opt['patch']//2:8]
                    g1_avg[:,:,count_raw] = img_low_g1[ii:opt['patch']//2:8,jj:opt['patch']//2:8]
                    g2_avg[:,:,count_raw] = img_low_g2[ii:opt['patch']//2:8,jj:opt['patch']//2:8]
                    b_avg[:,:,count_raw] = img_low_b[ii:opt['patch']//2:8,jj:opt['patch']//2:8]
                    count_raw=count_raw+1
        #             print('{},{},{}'.format(count_raw,ii,jj))

            
            gt.append(torch.from_numpy((np.transpose(img_gt_avg, [2, 0, 1]))).float())
            r_low.append(torch.from_numpy((np.transpose(r_avg, [2, 0, 1]))).float())
            g1_low.append(torch.from_numpy((np.transpose(g1_avg, [2, 0, 1]))).float())
            g2_low.append(torch.from_numpy((np.transpose(g2_avg, [2, 0, 1]))).float())
            b_low.append(torch.from_numpy((np.transpose(b_avg, [2, 0, 1]))).float())
            
        
        return gt, r_low, g1_low, g2_low, b_low, for_amplifier
    

obj_train = get_data(opt)
dataloader_train = DataLoader(obj_train, batch_size=opt['batch_size'], shuffle=opt['Shuffle'], num_workers=opt['workers'], pin_memory=opt['Pin_memory'])

for i,img in enumerate(dataloader_train):
    gt = img[0]
    r_low = img[1]
    g1_low = img[2]
    g2_low = img[3]
    b_low = img[4]
    for_amplifier = img[5]
    m = nn.PixelShuffle(8)

    imageio.imwrite('GT.jpg',resize(gt[0].reshape(-1,8,3,512//8,512//8).permute(2,3,0,4,1).reshape(1,3,512,512)[0].cpu().numpy().transpose(1,2,0)*255,(512,512)).astype(np.uint8))
    imageio.imwrite('ip.jpg',resize(m(g1_low[0])[0].cpu().numpy().transpose(1,2,0)*20*255,(512,512)).astype(np.uint8))
    
    
    break
    
print(r_low[0].size())
print(gt[0].size())
print(for_amplifier[0].size())




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
#             We SHALL DO the FINAL UNPACK 8x operation LATER IN THE CODE
        ) 
        
        
        
    def forward(self,r_low,g1_low,g2_low,b_low,for_amplifier):
        

        gamma = self.amplifier(for_amplifier)
        
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
        
        
        return alll,gamma
    
    
    
class common_functions():
    
    def __init__(self, opt):
        
        self.opt = opt
        self.count = 0
        self.relu = nn.ReLU(inplace=True)
                        
        self.device = torch.device("cuda")
        
        model = Net().apply(self.weights_init_kaiming)
#         checkpoint = torch.load(self.opt['dir_root']+'weights/'+self.opt['exp_name']+'_{}'.format(200000))
        #print(checkpoint)
        #model.load_state_dict(checkpoint['model'])
        print('Trainable parameters : {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        
        self.model = model.to(self.device)
        print(self.model)
        print(next(self.model.parameters()).is_cuda)  
        
        

        # define loss functions
        self.criterion = torch.nn.L1Loss()
        self.mseLoss = nn.MSELoss()

        # define optiizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt['lr'])
        #self.optimizer.load_state_dict(checkpoint['optimizer'])
        #self.optimizer_wraper = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, mode='max', factor=self.opt['gamma'], patience=self.opt['plateau_step'])
        #self.optimizer_wraper = torch.optim.lr_scheduler.StepLR(self.optimizer, self.opt['plateau_step'])
        self.optimizer.zero_grad()
        
        ##for gaussian kernel
        
        channels = 3
        gaussian_kernel = [[0.03797616, 0.044863533, 0.03797616],[0.044863533, 0.053, 0.044863533], [0.03797616, 0.044863533, 0.03797616]]
        
        gaussian_kernel = torch.from_numpy(np.asarray(gaussian_kernel)).float()

        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)


        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=3, groups=channels, bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False
        
        self.gaussian_filter = self.gaussian_filter.to(self.device)
        
        ### for perceptual loss
        
        model_ft = models.vgg19(pretrained=True)
        
        features = list(model_ft.features.children())[:19]
        
        features = nn.Sequential(*features)
        
        self.features = features.to(self.device)
        
        self.features = self.features.eval()
        
    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data)
        
    def vgg_features(self, x):
        results_out = []
        with torch.no_grad():
            for ii, model in enumerate(self.features):
                model.eval()
                x = model(x)
                if ii in {9,13,18}:
                    results_out.append(x)
        return results_out
    
    def perceptual_loss(self, img1, img2):
        
#         num = random.randint(0, 191)
#         num = (num//3)*3 
        
        dict1 = self.vgg_features(img1)
        dict2 = self.vgg_features(img2)
        
        w = np.asarray([ 0.8, 1, 1])
        
        loss_features = 0
        for i in range(len({9,13,18})):
            loss_features = loss_features + w[i]*self.criterion(dict1[i], dict2[i])
            
        return loss_features
        
        
    def blur_color_loss(self,gt,pred):
        
               
        
        gt = self.gaussian_filter(gt.detach())
        pred = self.gaussian_filter(pred)
        
#         plt.figure()
#         plt.imshow(gt[0,0:3,...].cpu().numpy().transpose(1,2,0))

        return self.mseLoss(pred,gt)
    

    def TVLoss(self,x,tv_weight=1):
    
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]),2).sum()

        tv_loss = tv_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

        _, c2, h2, w2 = x.size()
        chw2 = c2 * h2 * w2
        tv_loss = 1.0/chw2 * tv_loss
        
        return tv_loss


  
    
    def optimize_parameters(self,r_low,g1_low,g2_low,b_low,gt,for_amplifier):
        
#         num = random.randint(0, 191)
#         self.num = (num//3)*3 
        
        r_low=r_low.to(self.device)
        g1_low=g1_low.to(self.device)
        g2_low=g2_low.to(self.device)
        b_low=b_low.to(self.device)
        gt=gt.to(self.device)
        for_amplifier=for_amplifier.to(self.device)
        
        self.model.train()
        
        self.optimizer.zero_grad()
            
        pred_output, gamma = self.model(r_low,g1_low,g2_low,b_low,for_amplifier)
        
        # THIS IS THE UNPACK operation done using for loop so that readers can understand. The vectorised version of it which is faster can be found in the TESTING code.
        plot_out_GT = torch.zeros(1,3,512,512, dtype=torch.float).to(self.device)
        plot_out_pred = torch.zeros(1,3,512,512, dtype=torch.float).to(self.device)
        counttt=0
        for ii in range(8):
                for jj in range(8):

                    plot_out_GT[:,:,ii:opt['patch']:8,jj:self.opt['patch']:8] = gt[:,counttt:counttt+3,:,:]
                    plot_out_pred[:,:,ii:opt['patch']:8,jj:self.opt['patch']:8] = pred_output[:,counttt:counttt+3,:,:]
                    
                    counttt=counttt+3
        
        
        plot_out_pred = self.relu(plot_out_pred)
        
        blurLoss = self.criterion(plot_out_GT,plot_out_pred)
        
        
        
        regularization_loss = 0
        for param in self.model.parameters():
            regularization_loss += torch.sum(torch.abs(param))
            
        
        if self.count<self.opt['switch']:
            self.loss =  (blurLoss)  + (1e-6*regularization_loss) + (1000*self.TVLoss(plot_out_pred))  + (10*self.blur_color_loss(plot_out_GT.detach(),plot_out_pred)) + (3*self.perceptual_loss(plot_out_GT.detach(),plot_out_pred))
            if self.count%self.opt['text_prnt_freq']==0:
#                 print('TVLoss : {0: .6f}'.format(1000*self.TVLoss(pred_output)))
#                 print('L1loss MAIN : {0: .4f}'.format(5*blurLoss))
#                 print('ColorLoss : {0: .4f}'.format(10*self.blur_color_loss(imgs_op,pred_output)))
#                 print('reg_loss : {0: .4f}'.format(1e-6*regularization_loss))
#                 print('PerceptualLoss : {0: .4f}'.format(3*self.perceptual_loss(imgs_op,pred_output)))
                print('Count : {}\n'.format(self.count))
                print(gamma)
        else:
            self.loss =  (blurLoss)  + (1e-6*regularization_loss) + (400*self.TVLoss(plot_out_pred))  + (1*self.blur_color_loss(plot_out_GT.detach(),plot_out_pred)) + (3*self.perceptual_loss(plot_out_GT.detach(),plot_out_pred))
            if self.count%self.opt['text_prnt_freq']==0:
#                 print('TVLoss : {0: .6f}'.format(400*self.TVLoss(pred_output)))
#                 print('L1loss MAIN : {0: .4f}'.format(3*blurLoss))
#                 print('ColorLoss : {0: .4f}'.format(1*self.blur_color_loss(imgs_op,pred_output)))
#                 print('reg_loss : {0: .4f}'.format(1e-6*regularization_loss))
#                 print('PerceptualLoss : {0: .4f}'.format(3*self.perceptual_loss(imgs_op,pred_output)))
                print('Count : {}\n'.format(self.count))
                print(gamma)
            
        self.loss.backward()
        
        self.optimizer.step()
         
        self.count +=1
        if self.count%10==0:
            print(self.count)
            
            
        if self.count%opt['fig_freq']==0:
            
            plot_out_pred = (np.clip(plot_out_pred[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
            plot_out_GT = (np.clip(plot_out_GT[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
#             print(gt.shape)
#             print(np.dtype(pred_output))
#             counttt = 0
#             plot_out_GT = np.zeros((self.opt['patch'],self.opt['patch'],3),dtype=np.uint8)
#             plot_out_pred = np.zeros((self.opt['patch'],self.opt['patch'],3),dtype=np.uint8)
            
#             for ii in range(8):
#                 for jj in range(8):

#                     plot_out_GT[ii:opt['patch']:8,jj:self.opt['patch']:8,:] = gt[:,:,counttt:counttt+3]
#                     plot_out_pred[ii:opt['patch']:8,jj:self.opt['patch']:8,:] = pred_output[:,:,counttt:counttt+3]
                    
#                     counttt=counttt+3
            
            
            print("PSNR: {0:.3f}, SSIM: {1:.3f}, RMSE:{2:.3f}".format(PSNR(plot_out_GT,plot_out_pred), SSIM(plot_out_GT,plot_out_pred,multichannel=True),NRMSE(plot_out_GT,plot_out_pred)))
            
#             print('Input:')
#             plt.figure(figsize=(self.opt['fig_size'], self.opt['fig_size']))
                        
#             plt.imshow(plot_out_ip)
#             plt.show()
            
#             print('Predicted Output:')

            imageio.imwrite('pred_{}.jpg'.format(self.count), plot_out_pred)
            imageio.imwrite('GT_{}.jpg'.format(self.count), plot_out_GT)
                        
            
            
            print(gamma)
            
        if self.count in opt['save_freq']:
            torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }, self.opt['dir_root']+'weights/'+self.opt['exp_name']+'_{}'.format(self.count))

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    
    
    
    
gan_model = common_functions(opt)

for epoch in range(opt['epochs']):
    for iteration, img in enumerate(dataloader_train):
        gt = img[0]
        r_low = img[1]
        g1_low = img[2]
        g2_low = img[3]
        b_low = img[4]
        for_amplifier = img[5]
        
        for faster in range(4):
            gan_model.optimize_parameters(r_low[faster],g1_low[faster],g2_low[faster],b_low[faster],gt[faster],for_amplifier[faster])
                      
                
    
            


