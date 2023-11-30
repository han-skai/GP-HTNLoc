import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
torch.autograd.set_detect_anomaly(True)

import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import torch.optim
import torch.utils.data
import models
import data_got
import numpy as np
from utils.misc import AverageMeter

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils.metric import average_precision, hamming_loss, one_error, accuracy, ranking_loss
from graph_proto import get_f3_proto, get_a2_proto, get_f3_meta_graph, get_a2_meta_graph


parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('--model', default='./checkpoint/hsk_20230904_checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--num_sample', default=5, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
parser.add_argument('--lstm_hid_dim', default=150, type=int, metavar='N',
                    help='lstm_hid_dim')
parser.add_argument('--num_class', default=3, type=int, metavar='N',
                    help='the number of class')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--Bsamp_freq', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--Nsamp_freq', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--samp_freq', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--lamda', default=0.8, type=float,
                    metavar='N', help='the momentum decay of EMA')


def main():
    global args, best_micro
    args = parser.parse_args()

    base_transf,_,_ = data_got.one_sample_base2avg(batch_size=args.batch_size,sample_num=args.num_sample,samp_freq=args.Bsamp_freq,num_class=args.num_class)
    Ftest_loader, novel_loader,novelall_loader,test_y = data_got.Nload_data(batch_size=args.batch_size,sample_num=args.num_sample,samp_freq=args.Nsamp_freq, num_class=args.num_class, flag='EHTTN')
    model = models.Net(num_classes=args.num_class)

    print('==> Reading from model checkpoint..')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
          .format(args.model, checkpoint['epoch']))
    real_weight = model.classifier.fc.weight.data

    criterion3 = nn.MSELoss()
    trans_model3 = models.Transfer()
    optimizer3 = torch.optim.Adam(trans_model3.parameters(), lr=0.013, betas=(0.9, 0.99))

    for epoch in range(args.epochs):
        new_weight = get_proto(base_transf, model)
        train_loss, base_sum = train(trans_model3, criterion3, optimizer3, real_weight, new_weight)
        print("loss",train_loss)

    tail_weight = imprint(novel_loader, model, trans_model3, base_sum)

    model_criterion = nn.BCELoss()

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    output_all = []
    F1 = np.zeros(5)

    for i in range(args.Nsamp_freq):
        print("ensemble start!!!!!!!! this is calssifier ", i)
        model.classifier.fc.weight.data = tail_weight[i]
        fine_tuning(novelall_loader, model, model_criterion, model_optimizer)
        output = validate(Ftest_loader, model)
        output_all.append(output)
    output_all = (torch.sum(torch.tensor(output_all),0))/args.Nsamp_freq
    output_all[output_all > 0.5] = 1
    output_all[output_all <= 0.5] = 0
    for l in range(5):
        F1[l] = f1_score(test_y[:, l], output_all[:, l], average='binary')
    print("each class result f1!!!!!!!!!!!!!!")
    print(F1)

def get_proto(train_loader,model):
    base_rep = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(train_loader):
            output = model.extract(input)
            base_rep.extend(output.cpu().numpy())
    base_rep = np.array(base_rep)
    base_rep = torch.from_numpy(base_rep)

    new_weight = get_f3_meta_graph()
    return new_weight

def train(trans_model, criterion,optimizer3,real_weight,new_weight):
    trans_model.train()
    losses = []
    e = 0
    for h in range(args.Bsamp_freq):
        doc_avg = new_weight[e:e+args.num_class,:]
        e = e+args.num_class
        output = trans_model(doc_avg)
        loss3 = criterion(output, real_weight)
        losses.append(float(loss3))
        optimizer3.zero_grad()
        loss3.backward(retain_graph=True)
        optimizer3.step()
    avg_loss = np.mean(losses)
    base_sum = new_weight
    return avg_loss, base_sum



#  This function uses a trained converter to convert the new class from a prototype to a classifier parameter
#  (with attention added to the conversion process), and then splices it with the previous majority classifier
#  parameter to obtain the full classifier.
def imprint(novel_loader, model, trans_model, base_sum):
    attention=[]
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(novel_loader):
            output = model.extract(input)
            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)

    # new_weight = get_a2_proto()
    # new_weight = get_a2_proto_bi(output_stack)
    new_weight = get_a2_meta_graph()

    print("attention is all")
    tail_sampfre = []
    k = 0
    for i in range(args.samp_freq):
        new_weight_tow = new_weight[k:k+2]
        k = k + 2

        e = 0
        for h in range(args.samp_freq):
            doc_avg = base_sum[e:e + args.num_class, :]
            pro = F.softmax(torch.mm(new_weight_tow, doc_avg.t()), dim=1)
            new_rep = torch.mm(pro, doc_avg)
            attention.append(new_rep)
            e = e + args.num_class

        tail_corr = torch.zeros(2, 256)
        for m in range(args.samp_freq):
            tail_corr = tail_corr+attention[m]
        tail_corr = tail_corr/args.num_class

        tail_rep = (tail_corr+new_weight_tow)/2
        tail_real=trans_model.transfor(tail_rep)
        weight = torch.cat([model.classifier.fc.weight.data, tail_real])
        tail_sampfre.append(weight)
    print('imprint done')
    return tail_sampfre



#  The fine-tuning here is done by re-training a small number of the parameters
#  of the entire classifier on new classes after splicing them together to obtain the full classifier parameters.
def fine_tuning(train_loader, model, criterion, optimizer):
    print("+++++++++fine_tuning++++++++++++++++++++")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    microF1 = AverageMeter()
    macroF1 = AverageMeter()
    model.train()
    end = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg, microF1.avg, macroF1.avg)

def all_tuning(all_train, model, my_criterion, optimizer):
    print("+++++++++++++all_tuning++++++++++++++++++++")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    microF1 = AverageMeter()
    macroF1 = AverageMeter()
    model.train()
    end = time.time()
    for batch_idx, (input, target) in enumerate(all_train):
        data_time.update(time.time() - end)
        output = model(input)
        loss = my_criterion(output, target)
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, microF1.avg, macroF1.avg)


def validate(val_loader, model):
    one_iteration = []
    F1 = np.zeros(5)
    acc, ap, hl, oe, rl = 0, 0, 0, 0, 0
    model.eval()

    with torch.no_grad():

        for batch_idx, (input, target) in enumerate(val_loader):

            output = model(input)
            target = target.data.cpu().float()
            output = output.data.cpu()
            one_iteration.extend(output.numpy())
            ap += average_precision(output, target)
            oe += one_error(output, target)
            rl += ranking_loss(output, target)

            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            for l in range(5):
                F1[l] += f1_score(target[:, l], output[:, l], average='binary')
            acc += accuracy(output, target)
            hl += hamming_loss(output, target)

        print(len(val_loader))
        acc /= len(val_loader)
        ap /= len(val_loader)
        hl /= len(val_loader)
        oe /= len(val_loader)
        rl /= len(val_loader)
        np.set_printoptions(formatter={'float': '{: 0.4}'.format})
        print('the result of F1: \n', F1/len(val_loader))
        print("AP: %.4f , H_loss: %.4f , R_loss: %.4f , O_error: %.4f , ACC: %.4f " % (ap, hl, rl, oe, acc))
        return one_iteration



if __name__ == '__main__':
    main()