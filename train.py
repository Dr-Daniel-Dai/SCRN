import os
import shutil
import argparse

import torch
import torch.nn as nn

from net_sg import AlexNetPlusLatent

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler
output_file = open("./model_10/log_train.log", "a+")

parser = argparse.ArgumentParser(description='Deep Hashing')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--epoch', type=int, default=30, metavar='epoch',
                    help='epoch')
parser.add_argument('--pretrained', type=int, default=87.3, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=10, metavar='bts',
                    help='binary bits')
parser.add_argument('--path', type=str, default='model_10', metavar='P',
                    help='path directory')
args = parser.parse_args()

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
trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=25,
                                          shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='./data', train=False, download=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=25,shuffle=True, num_workers=0)   ##  num_workers=2

net = AlexNetPlusLatent(args.bits)
print(net)

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('hello_cuda')
    net=net.cuda()

softmaxloss = nn.CrossEntropyLoss().cuda()

optimizer4nn = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer4nn, milestones=[64], gamma=0.1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, outputs = net(inputs)
        loss = softmaxloss(outputs, targets)
        
        optimizer4nn.zero_grad()

        loss.backward()

        optimizer4nn.step()

        train_loss += softmaxloss(outputs, targets).item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
    return train_loss/(batch_idx+1)

def test_t():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ###   取消测试阶段的梯度，避免out of memory
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            _, outputs = net(inputs)
            loss = softmaxloss(outputs, targets)
            test_loss += loss.item()
    #        test_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
        test_acc_epoch = 100*int(correct) / int(total)
        print (test_acc_epoch)
        
    global best_acc
    if test_acc_epoch>best_acc:
        best_acc = test_acc_epoch
        
        print('Saving')
        if not os.path.isdir('{}'.format(args.path)):
            os.mkdir('{}'.format(args.path))
        torch.save(net.state_dict(), './{}/{}'.format(args.path, test_acc_epoch))
        outputlog=("epoch: %4.4g , best_acc: %4.4g \n"% ( epoch , best_acc ) )
        output_file.write(outputlog)
        
#    if epoch == args.epoch:
#        print('Saving')
#        if not os.path.isdir('{}'.format(args.path)):
#            os.mkdir('{}'.format(args.path))
#        torch.save(net.state_dict(), './{}/{}'.format(args.path, test_acc_epoch))

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, outputs = net(inputs)
        loss = softmaxloss(outputs, targets)
        test_loss += loss.item()
#        test_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
    acc = 100*int(correct) / int(total)
    print ( 'final_Acc: %.3f'%acc)
    
test_data = 0
if args.pretrained:
    net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.pretrained)))
    if test_data:
        test()
    else:
    #    if os.path.isdir('{}'.format(args.path)):
    #        shutil.rmtree('{}'.format(args.path))
        
        for epoch in range(start_epoch, start_epoch+args.epoch):
            print ('train_process(epoch_%s)'%epoch)
            train(epoch)
            print ('test_process(epoch_%s)'%epoch)
            test_t()
            scheduler.step()
        output_file.close()
else:
    #    if os.path.isdir('{}'.format(args.path)):
    #        shutil.rmtree('{}'.format(args.path))
        
        for epoch in range(start_epoch, start_epoch+args.epoch):
            print ('train_process(epoch_%s)'%epoch)
            train(epoch)
            print ('test_process(epoch_%s)'%epoch)
            test_t()
            scheduler.step()
        output_file.close()