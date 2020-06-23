import torch.nn as nn
import torch
from torchvision import models
from utils.bilinear_layers import *

net_model = models.vgg16(pretrained=True)    ###   alexnet

class PlusLatent(nn.Module):
    def __init__(self, bits, class_num):
        super(PlusLatent, self).__init__()
        
        self._is_all = True
        self.bits = bits
        self.class_num = class_num
#        print(net_model)  
        
        self.features_1 = nn.Sequential(*list(net_model.features.children())[:14])   ###  conv+pool
        self.features_2 = nn.Sequential(*list(net_model.features.children())[:21])   ###  conv+pool
        self.features_3 = nn.Sequential(*list(net_model.features.children())[:-2])   ###  conv+pool
        
        self.relu5_3 = nn.ReLU(inplace=True)
        self.avgpool_2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.avgpool_4 = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))
        
        self.conv = nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.matrix_sqrt = matrix_sqrt.apply
        self.sign_sqrt = sign_sqrt.apply
        self.classifier = nn.Linear(in_features=512 * 512, out_features=self.bits, bias=True)
        
        # self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()
        self.Linear2 = nn.Linear(self.bits, self.class_num)
        
        self.tau_ten = torch.eye(512)*0.00001
        
#        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, X):
        
        N = X.size()[0]
        if self._is_all:
            assert X.size() == (N, 3, 224, 224)
            
            X_1 = self.features_1(X)
#            print(X_1.size())
            X_2 = self.features_2(X)
#            print(X_2.size())
            X_3 = self.features_3(X)
#            print(X_3.size())
        
            features_1_ = self.avgpool_4(X_1)
            features_2_ = self.avgpool_2(X_2)
            
            features = torch.cat( [features_1_,features_2_,X_3], 1 )
            
            features = self.relu5_3(features)
            features_ =  self.conv(features)
        
        channel_num = 512
        assert features_.size() == (N, channel_num, 14, 14)
        
        X = self.relu5_3(features_)
        
        assert X.size() == (N, channel_num, 14, 14)
        X = torch.reshape(X, (N, channel_num, 14* 14))
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14* 14)
        assert X.size() == (N, channel_num, channel_num)
        
        ####   avoid nan

        X = X + self.tau_ten.view(1,channel_num,channel_num).repeat(20,1,1).type(torch.cuda.FloatTensor)  ## repeat(batch_size,1,1)
        
        X = self.matrix_sqrt(X)
        X = self.sign_sqrt(X)
        X = torch.reshape(X, (N, channel_num * channel_num))
        
#        batchSize = N
#        dim = X.data.shape[1]
#        dtype = X.dtype
#
#        I = torch.ones(dim,dim).triu().reshape(dim*dim)
#        index = I.nonzero()
#        y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = X.device).type(dtype)
#        y = X[:,index]

        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        # X = torch.sqrt(X + 1e-5)
        
        X_f = torch.nn.functional.normalize(X)
        
        X_f_2 = self.classifier(X_f)
        
        features = self.activation(X_f_2)
        
        result = self.Linear2(X_f_2)

        return features, result
