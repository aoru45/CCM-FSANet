import torch
import os
from tqdm import tqdm
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from ccm import CropAffine
from fsanet import FSANet
device = 0


from dataset import NormalDataset
def train(model,criterion,optimizer,dataloader,num_epoches = 200,scheduler = None):
    model["crop"].cuda(device)
    model["pred"].cuda(device)
    for epoch in range(num_epoches):
        for phase in ["eval"]:
            if phase == "train":
                running_loss = 0.0
                model["crop"].train()
                model["pred"].train()
                
                for inputs,inputs_rand,targets,loc in tqdm(dataloader[phase]):
                    
                    inputs = inputs.cuda(device)
                    inputs_rand = inputs_rand.cuda(device)
                    targets = targets.cuda(device)
                    
                    
                    pred_labels = model["pred"](inputs_rand)
                    loss_mae = criterion["mae"](pred_labels,targets)
                    
                    loss_pred = loss_mae
                    optimizer["pred"].zero_grad()
                    loss_pred.backward()

                    cropped_,pos = model["crop"](inputs)
                    pred_ = model["pred"](cropped_.detach())
                    loss_ = criterion["mae"](pred_,targets)
                    loss_.backward()
                    optimizer["pred"].step()
                    running_loss += loss_mae.item()
                    
                    loss_c = criterion["mae"](model["pred"](cropped_),targets)
                    
                    optimizer["crop"].zero_grad()
                    loss_c.backward()
                    optimizer["crop"].step()
                print("-epoch:{} -phase:{} -loss:{}".format(epoch,phase,running_loss/len(dataloader[phase])))
                if scheduler is not None:
                    scheduler.step()
            else:
                model["crop"].eval()
                model["pred"].eval()
                p = 0.0
                y = 0.0
                r = 0.0
                total = 0.0
                res = []
                with torch.no_grad():
                    for inputs,targets in tqdm(dataloader[phase]):
                        
                        inputs = inputs.cuda(device)
                        targets = targets.cuda(device)
                        
                        x,loc = model["crop"](inputs)
                        pred_labels = model["pred"](x) 

                        pred_pitch = pred_labels[:,0]
                        pred_yaw = pred_labels[:,1]
                        pred_roll = pred_labels[:,2]
                        pitch = targets[:,0]
                        yaw = targets[:,1]
                        roll = targets[:,2]
                        #total_loss = F.l1_loss(pred_labels,targets,reduction = "sum").item()
                        pitch_loss = F.l1_loss(pred_pitch,pitch,reduction = "sum").item()
                        yaw_loss = F.l1_loss(pred_yaw,yaw,reduction = "sum").item()
                        roll_loss = F.l1_loss(pred_roll,roll,reduction = "sum").item()
                        total_loss = (pitch_loss + yaw_loss + roll_loss)/3

                        p+=pitch_loss
                        y+=yaw_loss
                        r+=roll_loss
                        total += total_loss
                print("-epoch: {} -phase: {} -pitch: {} -yaw: {} -roll: {} -mae: {}".format(epoch,phase,p/len(dataloader[phase].dataset),y/len(dataloader[phase].dataset),r/len(dataloader[phase].dataset),total/len(dataloader[phase].dataset)))
                if epoch % 5 == 0:
                    torch.save(model["pred"].state_dict(),"./fsa-net/ckpt/fsa_{}_{}.pth".format("pred",epoch))
                    torch.save(model["crop"].state_dict(),"./fsa-net/ckpt/fsa_{}_{}.pth".format("crop",epoch))
                    save_image(x,"./fsa-net/tmp/{}.png".format(epoch))
if __name__ == "__main__":
    if not os.path.exists('fsa-net/ckpt'):
        os.makedirs('fsa-net/ckpt')
    if not os.path.exists('fsa-net/tmp'):
        os.makedirs('fsa-net/tmp')
    num_capsule = 3
    dim_capsule = 16
    routings = 2

    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
    
    model = {"crop":CropAffine(),"pred":FSANet(S_set)}
    criterion = {"mae":nn.L1Loss()}
    
    optimizerC = optim.Adam(model["crop"].parameters(),lr = 3e-3)
    optimizerP = optim.Adam(model["pred"].parameters(),lr = 1e-3)
    optimizer = {"crop":optimizerC,"pred":optimizerP}
    scheduler = optim.lr_scheduler.StepLR(optimizerP, step_size=15, gamma=0.1)
    ds_train = NormalDataset(dataset = "300w")
    ds_eval = NormalDataset(dataset = "aflw")
    train_dataloader = DataLoader(ds_train,128,shuffle = True,num_workers = 6,pin_memory=True)
    eval_dataloader = DataLoader(ds_eval,1,shuffle = False,num_workers = 6,pin_memory=True)
    dataloader = {"train":train_dataloader,"eval":eval_dataloader}
    train(model,criterion,optimizer,dataloader,scheduler = scheduler)

    