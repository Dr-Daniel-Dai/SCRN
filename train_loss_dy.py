import os
import shutil
import argparse
import loss_dy

import torch
import torch.nn as nn
from product_data import design_dy
import numpy as np
from net_sg import PlusLatent

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler
# output_file = open("./model_dy/log_train.txt", "a+")

parser = argparse.ArgumentParser(description='Deep Hashing')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')   ###   0.01
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--epoch', type=int, default=40, metavar='epoch',
                    help='epoch')
parser.add_argument('--pretrained', type=int, default=1.01, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=12, metavar='bts',
                    help='binary bits')
parser.add_argument('--path', type=str, default='model_dy/12', metavar='P',
                    help='path directory')
parser.add_argument('--alpha', type=float, default=10.0, help="loss parameter")
parser.add_argument('--class_num', type=int, default=14, help="positive negative pairs balance weight")
args = parser.parse_args()

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

# def CalcSim(batch_label, train_label):
#     S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
#     return S

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

best_acc = 0.0
start_epoch = 1


transform_train = transforms.Compose(
     [transforms.Resize(256),
       transforms.RandomCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
     [transforms.Resize(224),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_data = design_dy(root='./data', train=True, transform=transform_train)
test_data = design_dy(root='./data', train=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=0, pin_memory=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=False, num_workers=0, pin_memory=True)

##########################################################################################################

net = PlusLatent(args.bits,args.class_num)
print(net)

use_cuda = torch.cuda.is_available()

cross_loss = nn.CrossEntropyLoss().cuda()

if use_cuda:
    print('hello_cuda')
    net=net.cuda()

optimizer4nn = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer4nn, milestones=[20], gamma=0.1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
 
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            train_label_onehot = EncodingOnehot(targets, args.class_num)
            inputs, targets , train_label_onehot= inputs.cuda(), targets.cuda(), train_label_onehot.cuda()
        inputs, targets ,train_label_onehot = Variable(inputs),Variable(targets), Variable(train_label_onehot)
        
        # print (train_label_onehot)
        # print (targets)
        
        feature_fcout, outputs = net(inputs)
        
        cr_loss = cross_loss(outputs, targets)
        
#        s_loss = loss_dy.pairwise_loss(feature_fcout, train_label_onehot, alpha=args.alpha, class_num=args.class_num).cuda()
        
#        q_loss = loss_dy.quantization_loss(feature_fcout).cuda()

        co_loss = loss_dy.contrastive_loss(feature_fcout, train_label_onehot, margin = int (0.5 *args.class_num )).cuda()
        
#        loss = s_loss + 0.01 * q_loss + cr_loss
        
        loss = 0.5*cr_loss + 0.5*co_loss
        
        optimizer4nn.zero_grad()

        loss.backward()

        optimizer4nn.step()

        train_loss += loss.item()
        
        if np.isnan(loss.item()):
            break
        
        # train_s_loss += s_loss.item()
        # train_q_loss += q_loss.item()
        
        # _, predicted = torch.max(outputs.data, 1)
        # total += targets.size(0)
        # correct += predicted.eq(targets.data).cpu().sum()
        
        
        # print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
        print('epoch: %d '%epoch, batch_idx, len(trainloader), 'Cur_Loss: %.5f '%(loss.item()) ,'Ave_Loss: %.3f '% (train_loss/(batch_idx+1)))
        # optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)
        
        if epoch%5==0 and epoch != 0 and  batch_idx == len(trainloader)-1:
            print ('saving current model')
            torch.save(net.state_dict(), './{}/{}'.format(args.path, epoch))
        
    # writer.add_scalar('similarity loss', train_s_loss/(2000), epoch)
    # writer.add_scalar('quantization loss', train_q_loss/(2000), epoch)
    # writer.add_scalar('loss', train_loss/(2000), epoch)
        
    print ('epoch done')

    

if args.pretrained:
    net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.pretrained)))
        
    for epoch in range(start_epoch, start_epoch+args.epoch):
        print ('train_process(epoch_%s)'%epoch)
        train(epoch)

        scheduler.step()

else:
    #    if os.path.isdir('{}'.format(args.path)):
    #        shutil.rmtree('{}'.format(args.path))
        
    for epoch in range(start_epoch, start_epoch+args.epoch):
        print ('train_process(epoch_%s)'%epoch)
        train(epoch)

        scheduler.step()
