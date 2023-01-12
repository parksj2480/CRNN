import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchsummaryX import summary

## Baseline Model
class RecognitionModel(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=2048):
        super(RecognitionModel, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        
        # CNN Backbone = 사전학습된 resnet18 활용
        # https://arxiv.org/abs/1512.03385
        resnet = resnet18(pretrained=False)
        # 흑백 이미지 인풋을 받기위한 작업
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)

        # CNN Feature Extract
        resnet_modules = list(resnet.children())[:-3]
        self.feature_extract = nn.Sequential(
            *resnet_modules,
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.linear1 = nn.Linear(2304, rnn_hidden_size)
        
        # RNN
        self.rnn = nn.LSTM(input_size=rnn_hidden_size,
                            hidden_size=rnn_hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=2,
                            dropout=0.5)
        self.linear2 = nn.Linear(self.rnn_hidden_size*2, num_chars)
        
        
    def forward(self, x):
        # CNN
        x = self.feature_extract(x) # [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2) # [batch_size, width, channels, height]
        print(x.shape)
        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1) # [batch_size, T==width, num_features==channels*height]
        x = self.linear1(x)
        
        # RNN
        x, _ = self.rnn(x)
        x = self.linear2(x)
        output = x.permute(1, 0, 2) # [T, batch_size, num_classes==num_features]
        
        return output

if __name__ == '__main__':
    model = RecognitionModel(num_chars=2350).to('cpu')
    summary(model,torch.rand(16,1, 64, 224))
    input = torch.randn(3,1,64,224)
    print(model(input).shape)