import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle
import random

from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch_examples.utils import get_mnist_train_loader, get_svhn_train_loader, get_fashion_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader, get_svhn_test_loader, get_fashion_mnist_test_loader

from net import net_mnist, net_svhn, knn_cnn, knn_cnn_svhn

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import time
from collections import Counter
import shutil

from knn_attacks import FGSM, BIM

filename = "mnist_lenet5_clntrained.pt"
train_batch_size = 1024
test_batch_size = 1000
test_batch_size = 800
nb_epoch = 20
train = False
extract = False
# net = 'svhn'
# net = 'fashion_mnist'
net = 'mnist'


from advertorch.test_utils import LeNet5

class LeNet5_each_layer(LeNet5):

    def forward(self, x):
        out1 = self.maxpool1(self.relu1(self.conv1(x)))
        out2_ = self.maxpool2(self.relu2(self.conv2(out1)))
        out2 = out2_.view(out2_.size(0), -1)
        out3 = self.relu3(self.linear1(out2))
        out4 = self.linear2(out3)
        return out1, out2_, out3, out4

model_ = LeNet5()
model_.load_state_dict(
    torch.load(os.path.join('trained_models/', filename)))
model_.to(device)
model_.eval()

# get knn features
model1_ = LeNet5_each_layer()
model1_.load_state_dict(
    torch.load(os.path.join('trained_models/', filename)))
model1_.to(device)
model1_.eval()


print("net: "+net)

if net == 'mnist' or net == 'fashion_mnist':
    # get knn features
    model1 = net_mnist()
    model1.load_state_dict(
        torch.load(os.path.join(net, filename)))
    model1.to(device)
    model1.eval()
    model2 = knn_cnn()

elif net == 'svhn':
    model1 = net_svhn()
    model1.load_state_dict(torch.load('svhn/svhn_lenet5_clntrained.pt'))
    model1.to(device)
    model1.eval()
    model2 = knn_cnn_svhn()

if train:
    pretrained_dict = model1.state_dict()
    model_dict = model2.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model2.load_state_dict(model_dict)
    model2.to(device)
else:
    if net == 'mnist':
        model2.load_state_dict(torch.load('mnist/mnist_lenet5_knn.pt'))
    elif net == 'fashion_mnist':
        model2.load_state_dict(torch.load('fashion_mnist/svhn_knn.pt'))
    elif net == 'svhn':
        model2.load_state_dict(torch.load('svhn/svhn_knn.pt'))
    
    model2.to(device)
    model2.eval()


def extract_libary(model1, train_loader, net):
    net = 'Lenet5'
    features_trainloader = {}
    labels = []
    for l in range(4):
        features_trainloader[str(l+1)] = []
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        outputs = model1(data)
        for l in range(4):
            tmp_feat = outputs[l].view(-1).cpu().data.numpy()
            norm_feat = tmp_feat/np.linalg.norm(tmp_feat, ord=2, keepdims=True)
            features_trainloader[str(l+1)].append(norm_feat)
        labels.append(target.item())
    for l in range(4):
        features_trainloader[str(l+1)] = torch.tensor(features_trainloader[str(l+1)])

    pickle.dump(features_trainloader, open('features/features_trainloader_tensor_'+net+'.pickle', 'wb'), protocol=4)
    pickle.dump(labels, open('features/labels_'+net+'.pickle', 'wb'))

if extract:
    if net == 'mnist':
        libary_loader = get_mnist_train_loader(batch_size=1, shuffle=False)
    elif net == 'fashion_mnist':
        libary_loader = get_fashion_mnist_train_loader(batch_size=1, shuffle=False)
    elif net == 'svhn':
        libary_loader = get_svhn_train_loader(batch_size=1, shuffle=False)
    extract_libary(model1_, libary_loader, net)
    

with open('features/features_trainloader_tensor_'+net+'.pickle', 'rb') as handle:
    features_trainloader = pickle.load(handle)
with open('features/labels_'+net+'.pickle', 'rb') as handle:
    labels = pickle.load(handle)

for l in range(4):
    features_trainloader[str(l+1)] = features_trainloader[str(l+1)].to(device)


if net == 'mnist':
    train_loader = get_mnist_train_loader(batch_size=train_batch_size, shuffle=True)
    train_loader_save = get_mnist_train_loader(batch_size=train_batch_size, shuffle=False)
    test_loader = get_mnist_test_loader(batch_size=1, shuffle=False)
elif net == 'fashion_mnist':
    train_loader = get_fashion_mnist_train_loader(batch_size=train_batch_size, shuffle=True)
    test_loader = get_fashion_mnist_test_loader(batch_size=1, shuffle=False)
elif net == 'svhn':
    train_loader = get_svhn_train_loader(batch_size=train_batch_size, shuffle=True)
    test_loader = get_svhn_test_loader(batch_size=1, shuffle=False)



y_test = []
for i, (data, target) in enumerate(test_loader):
    y_test.append(target.item())
y_test = np.array(y_test)

np.random.seed(1234)
ind_cal = np.zeros((750, ), dtype=np.int32)
for i in range(10):
    ind = np.where(y_test == i)[0]
    np.random.shuffle(ind)
    ind_cal[i*75 : (i + 1)*75] = ind[:75]
ind_test = np.arange(len(y_test), dtype=np.int32)
ind_test = np.setdiff1d(ind_test, ind_cal)

if net == 'mnist':
    cal_loader = get_mnist_test_loader(batch_size=test_batch_size, shuffle=False, sub_idx=ind_cal)
    test_loader = get_mnist_test_loader(batch_size=test_batch_size, shuffle=False, sub_idx=ind_test)
elif net == 'fashion_mnist':
    cal_loader = get_fashion_mnist_test_loader(batch_size=test_batch_size, shuffle=False, sub_idx=ind_cal)
    test_loader = get_fashion_mnist_test_loader(batch_size=test_batch_size, shuffle=False, sub_idx=ind_test)
elif net == 'svhn':
    cal_loader = get_svhn_test_loader(batch_size=test_batch_size, shuffle=False, sub_idx=ind_cal)
    test_loader = get_svhn_test_loader(batch_size=test_batch_size, shuffle=False, sub_idx=ind_test)


def get_feats(model1, advs):
    feats = {}
    outputs = model1(advs)
    for i in range(4):
        feat = outputs[i].view(outputs[i].size(0), -1).data
        feats[str(i+1)] = feat
    return feats

def knn(ori, adv):
    ori_feat = ori
    adv_feat = adv
    adv_feat = adv_feat/torch.norm(adv_feat, p=2, dim=1, keepdim=True)
    sim = torch.mm(adv_feat, ori_feat.t())
    sim_sorted, idx = sim.sort(1, descending=True)
    return idx

setting_K = 75

def perturb(data):
    times = 1
    new_data = torch.zeros(times*data.shape[0], data.shape[1], data.shape[2], data.shape[3]).to(data.device)
    k = 0
    for i in range(len(data)):
        num = 0
        while num < times:
            num += 1
            j = i
            while j == i:
                j = random.randint(0, len(data)-1) 
            c = random.random()
            c = 1
            new_img = c*data[i] + (1-c)*data[j]
            new_data[k] = new_img
            k += 1
    return new_data

def count(knn_pre):
    frequency = [0 for i in range(10)]
    for i in knn_pre:
        frequency[i] += 1.0/len(knn_pre)
    # return frequency
    return frequency.index(max(frequency)) 

def save_knn(layer, j, knn_idxs, num_now, if_adv):
    for idx, i in enumerate(knn_idxs):
        ori_img_path = 'adversaries/ori_train/train/'+str(i.item()).zfill(5)+'.png'
        if if_adv==1:
            save_img_dir = 'adversaries/ori_train/knn_all_adv/'+str(num_now)+'/'+str(layer)
        elif if_adv==2:
            save_img_dir = 'adversaries/ori_train/knn_all/'+str(num_now)+'/'+str(layer)
        else:
            return
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        shutil.copy(ori_img_path, save_img_dir+'/'+str(idx)+'.png')


def get_perturb_data_label(data, model1, feats, labels, batch_idx, if_adv=0):
    data_perturb = perturb(data)
    features_cln = get_feats(model1, data_perturb)
    knn_preds = {}
    knn_preds_frequency = []
    for j in range(4):
        knn_preds[str(j+1)] = []
    for i in range(4):
        feat = features_cln[str(i+1)]
        knn_idxs = knn(feats[str(i+1)], feat)
        fre = np.zeros((len(feat), 10))
        for j in range(len(feat)):
            num_now = batch_idx*800+j
            knn_pre = np.array(labels)[knn_idxs[j][:setting_K].cpu().data.numpy()]

            # save_knn(i, j, knn_idxs[j][:setting_K], num_now, if_adv)
            
            frequency = Counter(knn_pre).most_common(1)[0][0]
            knn_preds[str(i+1)].append(frequency)
            
            fre[j] = np.array([setting_K-co for co in np.bincount(knn_pre, minlength=10)])
        knn_preds_frequency.append(fre)
    return data_perturb, knn_preds, np.array(knn_preds_frequency)


def draw_hist(sum_A, frequency, y_test, hist_name, l2_dis):
    p = np.zeros((len(frequency), 10))
    for fre_i, fre_s in enumerate(frequency):
        p[fre_i] = np.array([np.sum(ss <= sum_A) for ss in fre_s])/len(sum_A)
    print(np.mean(np.argmax(p, 1)==y_test))
    print('mean cred: '+str(np.mean(np.max(p, 1))))
    plt.hist(np.max(p, 1))
    plt.savefig(hist_name)
    plt.close()

    return np.argmax(p,1) != y_test

def attack_ori(model1, model2, test_loader, layer, features_trainloader, labels, sum_A, len_ind_test):
    model1.eval()

    tot_num = 0

    num_classify = 0 # classify acc
    num_adv_classify = 0

    num_knn_last_layer = 0
    num_knn = 0
    num_knn_adv_layer = 0
    num_knn_cnn = 0
    num_knn_adv = 0 # knn acc

    ori_cred = np.zeros((len_ind_test, 10))
    adv_cred = np.zeros((len_ind_test, 10))

    start = 0

    l2_dis = np.zeros(len(ind_test))

    for batch_idx, (data, target) in enumerate(test_loader):
        print(batch_idx)
        tot_num += len(data)
        data, target = data.to(device), target.to(device)

        outputs = model1(data)[-1]
        _, preds = torch.max(outputs, 1)
        num_classify += (preds==target).sum()


        outputs = model2(data)
        _, preds = torch.max(outputs[3+layer], 1)
        num_knn_cnn += (preds==target).sum() # knn_cnn

        data_perturb, target_knn, frequency = get_perturb_data_label(data, model1, features_trainloader, labels, batch_idx)

        num_knn_last_layer += (target.cpu().data.numpy()==np.array(target_knn['4'])).sum() # knn
        num_knn += (target.cpu().data.numpy()==np.array(target_knn[str(layer)])).sum() # knn
        ori_cred[start:start+len(data)] = np.sum(frequency, 0)
        print('ori FGSM')
        adv = FGSM(model1, data, target.cpu().data.numpy(), ori=True)
        # adv = BIM(model1, data, target.cpu().data.numpy(), ori=True)
        # np.save('adversaries/FGSM_ori/03/'+str(batch_idx)+'.npy', adv)

        outputs_adv = model1(torch.tensor(adv).type(data.type()).to(device))[-1]
        _, preds_adv = torch.max(outputs_adv, 1)
        num_adv_classify += (preds_adv==target).sum()

        _, target_knn_adv, frequency_adv = get_perturb_data_label(torch.tensor(adv).type(data.type()).to(device), model1, features_trainloader, labels, batch_idx)
        num_knn_adv += (target.cpu().data.numpy()==np.array(target_knn_adv['4'])).sum() 
        num_knn_adv_layer += (target.cpu().data.numpy()==np.array(target_knn_adv[str(layer)])).sum() 
        adv_cred[start:start+len(data)] = np.sum(frequency_adv, 0)
    

        l2_dis[start:start+len(data)] = np.sqrt(((adv-data.cpu().data.numpy())**2).sum(1).sum(1).sum(1))
        start += len(data)


    idx = draw_hist(sum_A, ori_cred, y_test[ind_test], 'ori_cred.png', l2_dis)
    idx_adv = draw_hist(sum_A, adv_cred, y_test[ind_test], 'adv_ori_cred.png', l2_dis)
    l2_dis_avg = (l2_dis[idx_adv].sum())/idx_adv.sum()/28/28*255

    print("tot: "+str(tot_num))
    print("num_knn: "+str(num_knn))
    print("num_knn_last_layer: "+str(num_knn_last_layer))
    print("num_knn_cnn: "+str(num_knn_cnn))
    print("num_knn_adv: "+str(num_knn_adv))
    print("num_knn_adv_layer: "+str(num_knn_adv_layer))
    print("clean sample num_classify"+str(num_classify))
    print("adv sample num_adv_classify: "+str(num_adv_classify))
    print("l2_dis_avg: "+str(l2_dis_avg))

layer = 3

# model1 = model1_
for batch_idx, (data, target) in enumerate(cal_loader):
    data, target = data.to(device), target.to(device)
    data_perturb, target_knn, frequency = get_perturb_data_label(data, model1, features_trainloader, labels, batch_idx)
A = []
for l in range(4):
    A_l = []
    for i in range(len(data)):
        A_l.append(frequency[l][i][target[i].item()])
    A.append(A_l)
A = np.array(A)
sum_A = np.sum(A, 0)

tot_num = 0
num_adv_classify = 0
num_knn_adv = 0
num_knn_adv_layer = 0

adv_cred = 4*setting_K*(np.ones((len(ind_test), 10)))

start = 0

l2_dis = 100*(np.ones(len(ind_test)))

test_ori = True
if test_ori:
    attack_ori(model1, model2, test_loader, layer, features_trainloader, labels, sum_A, len(ind_test))

for batch_idx, (data, target) in enumerate(test_loader):
    
    print(batch_idx)

    tot_num += len(data)
    data, target = data.to(device), target.to(device)
    data_perturb, target_knn, frequency = get_perturb_data_label(data, model1, features_trainloader, labels, batch_idx, if_adv=0)
    
    print("our FGSM")
    adv = FGSM(model2, data, target_knn, layer=layer, frequency=frequency)
    # adv = BIM(model2, data, target_knn, layer=layer, frequency=frequency)

    # adv on ori model
    outputs_adv = model1(torch.tensor(adv).type(data.type()).to(device))[-1]
    _, preds_adv = torch.max(outputs_adv, 1)
    num_adv_classify += (preds_adv==target).sum() # 1
    _, target_knn_adv, frequency_adv = get_perturb_data_label(torch.tensor(adv).type(data.type()).to(device), model1, features_trainloader, labels, batch_idx, if_adv=0)
    num_knn_adv += (target.cpu().data.numpy()==np.array(target_knn_adv['4'])).sum()  # 2
    num_knn_adv_layer += (target.cpu().data.numpy()==np.array(target_knn_adv['3'])).sum()  # 2

    adv_cred[start:start+len(data)] = np.sum(frequency_adv, 0) # 3

    l2_dis[start:start+len(data)] = np.sqrt(((adv-data.cpu().data.numpy())**2).sum(1).sum(1).sum(1))
    
    start += len(data)
    # break

idx_adv = draw_hist(sum_A, adv_cred, y_test[ind_test], 'adv_cred.png', l2_dis)
l2_dis_avg = (l2_dis[idx_adv].sum())/idx_adv.sum()/28/28*255

print("tot_num: "+str(tot_num))
print("num_adv_classify: "+str(num_adv_classify))
print("knn_adv: "+str(num_knn_adv))
print("knn_adv_layer: "+str(num_knn_adv_layer))
print("l2_dis_avg: "+str(l2_dis_avg))

    



