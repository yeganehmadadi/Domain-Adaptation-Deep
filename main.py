from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import data_loader
import resnet as models
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
acctotal = 0
count = 0
loss = 0
batch_size = 32
iteration = 2500
lr = 0.01
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "./dataset/"
source1_name = "amazon"
#source2_name = ""
target_name = "dslr"

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
#source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

def train(model):
    source1_iter = iter(source1_loader)
    #source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    iters  = []
    losses = []
    test_acc = []
    acc = []

    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        
        cls_loss, lr_loss, l1_loss, cls_loss_w = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        
        loss =  cls_loss_w + cls_loss + gamma * (lr_loss + l1_loss)
        loss.backward()
        optimizer.step()
        
        # my save the current training information
        iters.append(i)
        losses.append(float(loss)/batch_size) # compute *average* loss
        test_acc.append(test(model)) # compute training accuracy
    

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tlr_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), lr_loss.item(), l1_loss.item()))
                          
        

        if i % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            #print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
            print(source1_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

    # my plotting
    plt.plot(iters, losses)
    plt.title("Loss Curve (batch_size={}, lr={})".format(batch_size, lr))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(iters, test_acc)
    #plt.plot(iters, correct)
    plt.title("Accuracy Curve for Source={} and Target={} (batch_size={}, lr={})".format(source1_name, target_name, batch_size, lr))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()
        

            

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark = 0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred1 + pred2) / 2
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        #print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
        print('\nsource1 accnum {}'.format(correct1))

        acc = (100. * correct / len(target_test_loader.dataset))

        
    #return correct
    return acc




if __name__ == '__main__':
    model = models.NewModel(num_classes=31)
    print(model)
    if cuda:
        model.cuda()
    train(model)
    

