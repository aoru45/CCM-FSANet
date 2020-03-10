import torch
import torch.nn as nn
from torchvision.utils import save_image
import glob
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
class CropAffine(nn.Module):
    def __init__(self):
        super(CropAffine,self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(inplace = True),# 96
            nn.Conv2d(16,16,3,2,1),nn.BatchNorm2d(16),nn.ReLU(inplace = True),# 48
            nn.Conv2d(16,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 24
            nn.Conv2d(32,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 12
            nn.Conv2d(32,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 6
            nn.Conv2d(32,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(inplace = True),# 3
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
        m = torch.zeros((n,2,3),requires_grad = False)
        m[:,0,0] = boxW/w

        m[:,0,2] = (pos[:,0]/2 + pos[:,2]/2 - boxW/2) / w
        m[:,1,1] = boxH/h
        m[:,1,2] = (pos[:,1]/2 + pos[:,3]/2 - boxH/2) / h


        grid = F.affine_grid(m, x.size(),align_corners = True)
        return F.grid_sample(x, grid,align_corners = True),pos_return

if __name__ == "__main__":
    imgs = glob.glob("/media/xueaoru/DATA/ubuntu/head_pose/Mine/dataset/AFLW2000/*.jpg")
    imgs_path = imgs[-16:]
    crop_path = "../ckpt/crop_90.pth"
    cropnet = CropAffine()
    cropnet.load_state_dict(torch.load(crop_path,map_location="cpu"))
    cropnet.eval()
    
    transform = transforms.Compose(
        [
            transforms.Resize((64,64)),
                transforms.ToTensor(),
        ]
    )
    img_tensors = []
    cropped_tensors = []
    for img_path in imgs_path:
        img = Image.open(img_path)
        img_tensor = transform(img)
        img_tensors.append(img_tensor)
        cropped,_ = cropnet(img_tensor.unsqueeze(0))
        cropped_tensors.append(cropped)
    save_image(torch.stack(img_tensors,dim = 0),"before.png")
    save_image(torch.cat(cropped_tensors,dim = 0),"after.png")