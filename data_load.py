import random
import pandas as pd
import numpy as np
import os
from PIL import Image

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18
from torchvision import transforms

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ## Data Load & Train/Validation Split

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 길이가 1인 Data는 전부 훈련, 길이가 2이상인 데이터에서 80:20 분할.
def TrainValSplit(data_path,csv_name,random_state=41):
    df = pd.read_csv(data_path+csv_name)

    # 제공된 학습데이터 중 1글자 샘플들의 단어사전이 학습/테스트 데이터의 모든 글자를 담고 있으므로 학습 데이터로 우선 배치
    df['len'] = df['label'].str.len()
    train_v1 = df[df['len']==1]

    # 제공된 학습데이터 중 2글자 이상의 샘플들에 대해서 단어길이를 고려하여 Train (80%) / Validation (20%) 분할
    df = df[df['len']>1]
    train_v2, val_df, _, _ = train_test_split(df, df['len'], test_size=0.2, random_state=random_state)

    # 학습 데이터로 우선 배치한 1글자 샘플들과 분할된 2글자 이상의 학습 샘플을 concat하여 최종 학습 데이터로 사용
    train_df = pd.concat([train_v1, train_v2])
    print("%d개의 Train Set과 %d개의 Validation Set으로 분할되었습니다." % (len(train_df), len(val_df)))
    
    # 학습 데이터로부터 단어 사전(Vocabulary) 구축
    train_gt = [gt for gt in train_df['label']]
    train_gt = "".join(train_gt)
    letters = sorted(list(set(list(train_gt))))
    print(len(letters))

    vocabulary = ["-"] + letters
    print("총 %d개의 단어사전 생성." % len(vocabulary))
    idx2char = {k:v for k,v in enumerate(vocabulary, start=0)}
    char2idx = {v:k for k,v in idx2char.items()}
    
    return (train_df,val_df,idx2char,char2idx)


## Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, img_width, img_height, train_mode=True, augmentation_mode=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.train_mode = train_mode
        self.img_width = img_width
        self.img_height = img_height
        self.augmentation_mode = augmentation_mode
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        image = Image.open(self.img_path_list[index]).convert('L') # 사진을 흑백으로 저장 color를 원하면 'RGB'
        
        # Padding 을 위한 사전작업
        w,h = image.size
        pw=int((self.img_width-w)/2)
        self.padding = (pw, 0, pw, 0)
        
        if self.train_mode:
            if not self.augmentation_mode:
                image = self.train_transform(image)
            else:
                image = self.aug_transform(image)
        else:
            image = self.test_transform(image)
            
        if self.label_list is not None:
            text = self.label_list[index]
            return image, text
        else:
            return image
        
    # Image Augmentation
    def train_transform(self, image):
        transform_ops = transforms.Compose([
            #transforms.Pad(self.padding,padding_mode='edge'),
            #transforms.Resize((self.img_height,self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.45,), std=(0.225,))
        ])
        return transform_ops(image)
    
    def test_transform(self, image):
        transform_ops = transforms.Compose([
            #transforms.Pad(self.padding,padding_mode='edge'),
            #transforms.Resize((self.img_height,self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.45,), std=(0.225,))
        ])
        return transform_ops(image)

    def aug_transform(self, image):
        kernel_size = random.randrange(5,20,2)
        transform_ops = transforms.Compose([
            transforms.RandomAffine(1,shear=40), # Randomly shear
            #transforms.RandomAdjustSharpness(10,p=0.1), # Randomly change sharpeness
            transforms.RandomRotation(degrees=30), # Randomly rotate the image
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), # Randomly change the brightness, contrast, saturation, and hue of the image
            transforms.GaussianBlur(kernel_size=kernel_size,sigma=(1,15)),
            #transforms.Pad(self.padding,padding_mode='edge'),
            #transforms.Resize((self.img_height,self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.45,),std=(0.225,)),
            #transforms.RandomErasing()
        ])
        return transform_ops(image)
