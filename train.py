import random
import pandas as pd
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18
from torchvision import transforms

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore') 

from data_load import seed_everything
from data_load import CustomDataset
from data_load import TrainValSplit

from utils import remove_duplicates
from utils import correct_prediction

from model import RecognitionModel


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Device chosen : ",device)

data_path='../../data/'
csv_name='CUTMIX_train.csv'

## Hyperparameter Setting
global CFG
CFG = {
    'IMG_HEIGHT_SIZE':64,
    'IMG_WIDTH_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':128,
    'NUM_WORKERS':4, # 본인의 GPU, CPU 환경에 맞게 설정
    'SEED':41,
    'augmentation':3
}


# ## Fixed RandomSeed
seed_everything(CFG['SEED']) # Seed 고정


## Data Load & Train/Validation Split
train_df, val_df, idx2char, char2idx = TrainValSplit(data_path,csv_name,CFG['SEED']) # [train val dataframe] , [단어사전 Dictionary] 반환

## CustomDataset 불러오기
train_dataset = CustomDataset(data_path+train_df['img_path'].str[1:].values, train_df['label'].values, img_width=CFG['IMG_WIDTH_SIZE'], img_height=CFG['IMG_HEIGHT_SIZE'], train_mode=True)

## AugmentDataset 만들기
augment_dataset_list=[]
for i in range(CFG['augmentation']):
    augment_dataset_list.append(CustomDataset(data_path+train_df['img_path'].str[1:].values, train_df['label'].values, img_width=CFG['IMG_WIDTH_SIZE'], img_height=CFG['IMG_HEIGHT_SIZE'], train_mode=True,augmentation_mode=True))
augment_dataset=torch.utils.data.ConcatDataset(augment_dataset_list)
train_dataset = torch.utils.data.ConcatDataset([train_dataset, augment_dataset])
print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])

val_dataset = CustomDataset(data_path+val_df['img_path'].str[1:].values, val_df['label'].values, img_width=CFG['IMG_WIDTH_SIZE'], img_height=CFG['IMG_HEIGHT_SIZE'], train_mode=False)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])


## Define CTC Loss
criterion = nn.CTCLoss(blank=0) # idx 0 : '-'

## Train
best_model_num=0

def train(model, optimizer, scheduler, device):
    model.to(device)
    global best_model_num
    best_loss = 999999
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        
        for image_batch, text_batch in tqdm(iter(train_loader)):
            image_batch = image_batch.to(device)
            
            optimizer.zero_grad()
            
            text_batch_logits = model(image_batch)
            loss = compute_loss(text_batch, text_batch_logits)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        _train_loss = np.mean(train_loss)
        
        _val_loss = validation(model, val_loader, device)
        print(f'Epoch : [{epoch}] Train CTC Loss : [{_train_loss:.5f}] Val CTC Loss : [{_val_loss:.5f}]')
        
        if scheduler is not None:
            scheduler.step(_val_loss)
        
        if best_loss > _val_loss:
            best_loss = _val_loss
            best_model = model
            best_model_num=epoch
    return best_model


## Validation

def validation(model, val_loader, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for image_batch, text_batch in tqdm(iter(val_loader)):
            image_batch = image_batch.to(device)
            
            text_batch_logits = model(image_batch)
            loss = compute_loss(text_batch, text_batch_logits)
            
            val_loss.append(loss.item())
    
    _val_loss = np.mean(val_loss)
    return _val_loss



## Utils

def encode_text_batch(text_batch):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)
    
    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)
    
    return text_batch_targets, text_batch_targets_lens
  
  
  
  
def compute_loss(text_batch, text_batch_logits):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(device) # [batch_size]

    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss

def decode_predictions(text_batch_logits):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for image_batch in tqdm(iter(test_loader)):
            image_batch = image_batch.to(device)
            
            text_batch_logits = model(image_batch)
            
            text_batch_pred = decode_predictions(text_batch_logits.cpu())
            
            preds.extend(text_batch_pred)
    return preds
    

######## Run!! #############
if __name__ == '__main__':
    ## Model Define
    model=RecognitionModel(num_chars=len(char2idx)) # 단어 사전 갯수 정의
    model.load_state_dict(torch.load('model.pth'))

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

    model.eval()
    infer_model = train(model, optimizer, scheduler, device)


    ## Inference
    test = pd.read_csv(data_path+'test.csv')

    test_dataset = CustomDataset(data_path+test['img_path'].str[1:].values, None,img_width=CFG['IMG_WIDTH_SIZE'], img_height=CFG['IMG_HEIGHT_SIZE'], train_mode=False)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

    predictions = inference(infer_model, test_loader, device)


    # ## Submission

    submit = pd.read_csv(data_path+'sample_submission.csv')
    submit['temp_label'] = predictions
    submit['label'] = submit['temp_label'].apply(correct_prediction)

    print("%d번째 epoch 모델이 저장됨." % best_model_num)
    submit.to_csv('./submission0.csv', index=False)
    torch.save(infer_model.state_dict(),'./model0.pth')
