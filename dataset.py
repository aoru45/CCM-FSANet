
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io as sio
import numpy as np
import torch
from PIL import Image
class NormalDataset(Dataset):
    def __init__(self,ds_path = "../dataset", dataset = "300w"):
        self.ds_name = dataset
        if dataset == "300w":
            self.imgs = []
            sub_dirs = ["AFW","AFW_Flip","HELEN","HELEN_Flip","IBUG","IBUG_Flip","LFPW","LFPW_Flip"]
            #sub_dirs = ["AFW"]
            for dir in sub_dirs:
                self.imgs.extend(glob.glob("{}/{}/{}/*.jpg".format(ds_path,"300W_LP",dir)))
            self.transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.RandomErasing()
                ])
            self.transform_rand = transforms.Compose([
                transforms.Resize((96,96)),
                transforms.RandomCrop(64),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.RandomErasing(),
                ])
        else:
            self.imgs = glob.glob("{}/{}/*.jpg".format(ds_path,"AFLW2000"))
            self.transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                ])
    def __getitem__(self,idx):
        img_path = self.imgs[idx]
        mat_path = img_path.replace(".jpg",".mat")
        mat_contents = sio.loadmat(mat_path)
        pose_para = mat_contents['Pose_Para'][0]
        pt2d = mat_contents['pt2d']

        pt2d_x = pt2d[0,:]
        pt2d_y = pt2d[1,:]
        pt2d_idx = pt2d_x>0.0
        pt2d_idy= pt2d_y>0.0

        pt2d_id = pt2d_idx
        if sum(pt2d_idx) > sum(pt2d_idy):
            pt2d_id = pt2d_idy

        pt2d_x = pt2d_x[pt2d_id]
        pt2d_y = pt2d_y[pt2d_id]

        #img = cv.imread(img_path)
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Crop the face loosely
        x_min_ = min(pt2d_x)
        x_max_ = max(pt2d_x)
        y_min_ = min(pt2d_y)
        y_max_ = max(pt2d_y)
        # the original
        x_min = int(min(pt2d_x))
        x_max = int(max(pt2d_x)) 
        y_min = int(min(pt2d_y))
        y_max = int(max(pt2d_y))
        
        h = y_max-y_min
        w = x_max-x_min

        #ad = 0.8
        #x_min = max(int(x_min - ad * w), 0)
        #x_max = min(int(x_max + ad * w), img_w - 1)
        #y_min = max(int(y_min - ad * h), 0)
        #y_max = min(int(y_max + ad * h), img_h - 1)
        #img = img.crop([x_min,y_min,x_max,y_max])
        #h = y_max - y_min
        #w = x_max - x_min
        #x_min_ -= x_min
        #x_max_ -= x_min
        #y_min_ -= y_min
        #y_max_ -= y_min
        #assert y_max > y_max_ and x_max > x_max_
        #img = img[y_min:y_max,x_min:x_max]


        pitch = pose_para[0] * 180 / np.pi
        yaw = pose_para[1] * 180 / np.pi
        roll = pose_para[2] * 180 / np.pi

        #img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        label = torch.FloatTensor([pitch,yaw,roll])
        face_label = torch.tensor([x_min_/img_w,y_min_/img_h,x_max_/img_w,y_max_/img_h])
        if self.ds_name == "300w":
            return self.transform(img),self.transform_rand(img), label, face_label
        else:
            return self.transform(img),label
    def __len__(self,):
        return len(self.imgs)
class BIWIDataset(Dataset):
    def __init__(self,ds_path = "./dataset", dataset = "biwi"):
        self.ds_name = dataset
        dataset = np.load("/media/xueaoru/DATA/ubuntu/head_pose/fsa-net/data/biwi.npz")
        self.imgs = dataset["image"]
        self.pose = dataset["pose"]

        self.transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                ])
    def __getitem__(self,idx):
        img_cv = np.array(self.imgs[idx])
        labels = np.array(self.pose[idx])
        yaw, pitch, roll = labels[0], labels[1] , labels[2]
        img = Image.fromarray(img_cv)
        return self.transform(img), torch.FloatTensor([pitch,yaw,roll])
    def __len__(self):
        return self.imgs.shape[0]
class Pointing04Dataset(Dataset):
    def __init__(self):
        dataset = np.load("./dataset/pointing04.npz")
        self.imgs = dataset["image"]
        self.pose = dataset["pose"]

        self.transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                ])
    def __getitem__(self,idx):
        img_cv = np.array(self.imgs[idx])
        labels = np.array(self.pose[idx])
        pitch, yaw = labels[0], labels[1]
        img = Image.fromarray(img_cv)
        return self.transform(img), torch.FloatTensor([pitch,yaw])
    def __len__(self):
        return self.imgs.shape[0]