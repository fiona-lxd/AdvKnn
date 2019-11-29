import torch
import torch.nn.functional as F
import numpy as np
import pdb

from torch.autograd import Variable
def tensor2variable(x=None, device=None, requires_grad=False):
    """
    :param x:
    :param device:
    :param requires_grad:
    :return:
    """
    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)

def FGSM(model2, data, target, layer=4, frequency=None, ori=False):
    device = data.device
    samples = data.cpu().data.numpy()
    copy_samples = np.copy(samples)
    var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
    # pdb.set_trace()
    if isinstance(layer, int):
        if ori:
            var_ys = tensor2variable(torch.LongTensor(np.array(target)), device=device)
        else:
            var_ys = tensor2variable(torch.LongTensor(np.array(target[str(layer)])), device=device)

        if ori:
            preds = model2(var_samples)[-1]
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, var_ys)
        
        else:
            preds = model2(var_samples)[layer-5]
            criterion1 = torch.nn.CrossEntropyLoss()
            criterion2 = torch.nn.KLDivLoss(reduce=True)
            loss1 = criterion1(preds, var_ys)
            loss_KL = criterion2(F.log_softmax(preds,1), torch.tensor((75-frequency[layer-1])/75.0).type(preds.type()))
            # loss = loss_KL + 20*loss1
            loss = loss1
    else:
        var_ys = {}

        var_ys['1'] = tensor2variable(torch.LongTensor(np.array(target['1'])), device=device)
        var_ys['2'] = tensor2variable(torch.LongTensor(np.array(target['2'])), device=device)
        var_ys['3'] = tensor2variable(torch.LongTensor(np.array(target['3'])), device=device)
        var_ys['4'] = tensor2variable(torch.LongTensor(np.array(target['4'])), device=device)
        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.KLDivLoss(reduce=True)
        preds = model2(var_samples)
        loss = 0
        for l in range(4):
            loss1 = criterion1(preds[-1-l], var_ys[str(l+1)])
            loss_KL = criterion2(F.log_softmax(preds[-1-l],1), torch.tensor((75-frequency[l])/75.0).type(preds[-1].type()))
            loss += (loss_KL + 0.3*loss1)


    print(loss)
    loss.backward()
    gradient_sign = var_samples.grad.data.cpu().sign().numpy()

    adv_samples = copy_samples + 0.25 * gradient_sign
    adv_samples = np.clip(adv_samples, 0.0, 1.0)

    return adv_samples


def BIM(model2, data, target, layer=4, frequency=None, ori=False):
    device = data.device
    samples = data.cpu().data.numpy()
    copy_samples = np.copy(samples)
    
    num_steps = 100
    epsilon = 0.25
    epsilon_iter = 0.01

    for index in range(num_steps):
        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)

        model2.eval()

        if isinstance(layer, int):
            if ori:
                var_ys = tensor2variable(torch.LongTensor(np.array(target)), device=device)
            else:
                var_ys = tensor2variable(torch.LongTensor(np.array(target[str(layer)])), device=device)

            if ori:
                preds = model2(var_samples)[-1]
                loss_fun = torch.nn.CrossEntropyLoss()
                loss = loss_fun(preds, var_ys)
        
            else:
                preds = model2(var_samples)[layer-5]
                criterion1 = torch.nn.CrossEntropyLoss()
                criterion2 = torch.nn.KLDivLoss(reduce=True)
                loss1 = criterion1(preds, var_ys)
                loss_KL = criterion2(F.log_softmax(preds,1), torch.tensor((75-frequency[layer-1])/75.0).type(preds.type()))
                loss = loss_KL + 20*loss1
                # loss = loss1
        else:
            var_ys = {}

            var_ys['1'] = tensor2variable(torch.LongTensor(np.array(target['1'])), device=device)
            var_ys['2'] = tensor2variable(torch.LongTensor(np.array(target['2'])), device=device)
            var_ys['3'] = tensor2variable(torch.LongTensor(np.array(target['3'])), device=device)
            var_ys['4'] = tensor2variable(torch.LongTensor(np.array(target['4'])), device=device)
            criterion1 = torch.nn.CrossEntropyLoss()
            criterion2 = torch.nn.KLDivLoss(reduce=True)
            preds = model2(var_samples)
            loss = 0
            for l in range(4):
                loss1 = criterion1(preds[-1-l], var_ys[str(l+1)])
                loss_KL = criterion2(F.log_softmax(preds[-1-l],1), torch.tensor((75-frequency[l])/75.0).type(preds[-1].type()))
                loss += (loss_KL + 0.3*loss1)

        loss.backward()

        gradient_sign = var_samples.grad.data.cpu().sign().numpy()
        copy_samples = copy_samples + epsilon_iter * gradient_sign
        copy_samples = np.clip(copy_samples, samples - epsilon, samples + epsilon)
        copy_samples = np.clip(copy_samples, 0.0, 1.0)
    return copy_samples


