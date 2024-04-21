import argparse
import os
import shutil
# import time
# from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import numpy as np
import data_got


from utils.logger import Logger, savefig
from utils.misc import AverageMeter, mkdir_p
from utils.metric import average_precision, hamming_loss, one_error, accuracy, ranking_loss


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-c', '--checkpoint', default='./checkpoint/snoRNA_merge_124_01_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: pretrain_checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--lstm_hid_dim', default=150, type=int, metavar='N',
#                     help='lstm_hid_dim')
parser.add_argument('--num_class', default=3, type=int, metavar='N',
                    help='the number of class')
parser.add_argument('--num-sample', default=5, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--lamda', default=0.5, type=float,
                    metavar='N', help='the momentum decay of EMA')

best_micro = 0

def main():
    global args, best_micro
    args = parser.parse_args()
    train_loader, val_loader, _ = data_got.Bload_data(batch_size=args.batch_size, num_class=args.num_class)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.Net(args.num_class)

    criterion = nn.BCELoss()
    extractor_params = list(map(id, model.extractor.parameters()))
    classifier_params = filter(lambda p: id(p) not in extractor_params, model.parameters())
    optimizer = torch.optim.Adam([
                {'params': model.extractor.parameters()},
                {'params': classifier_params, 'lr': args.lr * 10}
            ], lr=args.lr,  betas=(0.9, 0.99))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_micro = checkpoint['best_micro']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        lr = optimizer.param_groups[1]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        print(f'{epoch}-th epoch，train_loss: {train_loss}')
        micro = validate(val_loader, model, criterion)
        val_micro = micro[2]

        is_best = val_micro > best_micro
        best_micro = max(val_micro, best_micro)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_micro': best_micro,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        if is_best == 1:
            best_epoch = epoch+1

    savefig(os.path.join(args.checkpoint, 'log.eps'))
    print('Best val_micro:%.6f ,%d-th epoch get best' % (best_micro, best_epoch))



def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()

    for batch_idx, (input, target) in enumerate(train_loader):

        output = model(input)
        loss = criterion(output, target.float())
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    score_micro = np.zeros(3)
    test_p1, test_p2, test_p3 = 0, 0, 0
    test_ndcg1, test_ndcg2, test_ndcg3 = 0, 0, 0
    acc, ap, hl, oe, rl = 0, 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):

            output = model(input)
            loss = criterion(output, target.float())
            target = target.data.cpu().float()
            output = output.data.cpu()

            ap += average_precision(output, target)
            oe += one_error(output, target)
            rl += ranking_loss(output, target)


            output[output > 0.5] = 1
            output[output <= 0.5] = 0

            acc += accuracy(output, target)
            hl += hamming_loss(output, target)

            score_micro += [precision_score(target, output, average='micro'),
                            recall_score(target, output, average='micro'),
                            f1_score(target, output, average='micro')]

            losses.update(loss.item(), input.size(0))


        acc /= len(val_loader)
        ap /= len(val_loader)
        hl /= len(val_loader)
        oe /= len(val_loader)
        rl /= len(val_loader)

        print("AP: %.4f , H_loss: %.4f , R_loss: %.4f , O_error: %.4f , ACC: %.4f " % (ap, hl, rl, oe, acc))
        print(f'validate_loss: {loss}')

        return score_micro / len(val_loader)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
