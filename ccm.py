import torch
import torch.nn as nn
import torch.nn.functional as F
device = 0
class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
class CropAffine(nn.Module):
    def __init__(self):
        super(CropAffine,self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace = True),# 
            nn.Conv2d(16,16,3,2,1),nn.BatchNorm2d(16),nn.ReLU(inplace = True),# 
            nn.Conv2d(16,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 
            nn.Conv2d(32,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 
            nn.Conv2d(32,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 
            nn.Conv2d(32,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 
        )
        self.loc = nn.Sequential(
            nn.Linear(32*2*2,16),
            nn.Linear(16, 4),
            nn.Sigmoid()
            )
        
        self.loc[1].weight.data.zero_()
        self.loc[1].bias.data.copy_(torch.tensor([-2.,-2., 2.,2.], dtype=torch.float))
        #self.crop = AttentionCropLayer()
    def forward(self,x,):
        n,_,h,w = x.size()
        feature = self.feature_extractor(x)
        pos = self.loc(feature.view(-1,32*2*2))
        pos_return = pos.clone()
        #croped = self.crop(x,pos)
        #return croped,pos
        
        pos = pos * h
        boxW, boxH = pos[:,2] - pos[:,0],pos[:,3] - pos[:,1]
        m = torch.zeros((n,2,3),requires_grad = False).cuda(device)
        m[:,0,0] = boxW/w

        m[:,0,2] = (pos[:,0]/2 + pos[:,2]/2 - boxW/2) / w
        m[:,1,1] = boxH/h
        m[:,1,2] = (pos[:,1]/2 + pos[:,3]/2 - boxH/2) / h


        grid = F.affine_grid(m, x.size(),align_corners = True)
        return F.grid_sample(x, grid,align_corners = True),pos_return
class CropAffineAlias(nn.Module):
    def __init__(self):
        super(CropAffineAlias,self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace = True),# 96
            nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace = True),# 48
            nn.MaxPool2d(kernel_size=2, stride=1),
            Downsample(channels=16, filt_size=3, stride=2),
            nn.Conv2d(16,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 24
            nn.MaxPool2d(kernel_size=2, stride=1),
            Downsample(channels=32, filt_size=3, stride=2),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 12
            nn.MaxPool2d(kernel_size=2, stride=1),
            Downsample(channels=32, filt_size=3, stride=2),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 6
            nn.MaxPool2d(kernel_size=2, stride=1),
            Downsample(channels=32, filt_size=3, stride=2),
            nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 3
            nn.MaxPool2d(kernel_size=2, stride=1),
            Downsample(channels=32, filt_size=3, stride=2),
        )
        self.loc = nn.Sequential(
            nn.Linear(32*2*2,16),
            nn.Linear(16, 4),
            nn.Sigmoid()
            )
        
        self.loc[1].weight.data.zero_()
        self.loc[1].bias.data.copy_(torch.tensor([-2.,-2., 2.,2.], dtype=torch.float))
        #self.crop = AttentionCropLayer()
    def forward(self,x,):
        n,_,h,w = x.size()
        feature = self.feature_extractor(x)
        pos = self.loc(feature.view(-1,32*2*2))
        pos_return = pos.clone()
        #croped = self.crop(x,pos)
        #return croped,pos
        
        boxW, boxH = pos[:,2] - pos[:,0],pos[:,3] - pos[:,1]
        m = torch.zeros((n,2,3),requires_grad = False)
        m[:,0,0] = boxW

        m[:,0,2] = (pos[:,0]/2 + pos[:,2]/2 - boxW/2)
        m[:,1,1] = boxH
        m[:,1,2] = (pos[:,1]/2 + pos[:,3]/2 - boxH/2) 


        grid = F.affine_grid(m, x.size(),align_corners = True)
        return F.grid_sample(x, grid,align_corners = True),pos_return