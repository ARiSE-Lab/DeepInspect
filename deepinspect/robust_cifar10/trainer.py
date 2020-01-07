from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
#from convex_adversarial import robust_loss, robust_loss_parallel
import torch.optim as optim

import numpy as np
import time
import gc

from attacks import _pgd

DEBUG = False

'''
def evaluate_robust_new(loader, model, evaluate_size, epsilon, epoch, log, verbose, 
                    real_time=False, parallel=False, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()

    end = time.time()

    torch.set_grad_enabled(False)
    for i, (X,y) in enumerate(loader):
        if(i>=evaluate_size/y.shape[0]):
            break
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        m = y.shape[0]
        rc = 0.0
        re = 0.0
        for r in range(m):
            robust_ce, robust_err = robust_loss(model, epsilon, X[r:r+1], y[r:r+1], **kwargs)
            rc+=robust_ce
            re+=robust_err
        robust_ce = rc/float(m)
        robust_err = re/float(m)

        # _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.item(), robust_err, ce.item(), err.item(),
           file=log)
        if verbose: 
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, robust_ce, out, ce
        if DEBUG and i ==10: 
            break
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    return robust_errors.avg



'''

def evaluate_baseline(loader, model, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    loader = list(loader)
    end = time.time()
    for i, (X,y) in enumerate(loader):
        if(i>1024/y.shape[0]):
            break
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        #print (i, err)

        # print to logfile
        print(epoch, i, ce.data[0], err, file=log)

        # measure accuracy and record loss
        losses.update(ce.data[0], X.size(0))
        errors.update(err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % verbose == 0: 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))
        log.flush()

    print(' * Error {error.avg:.3f}'
          .format(error=errors))
    return errors.avg





def evaluate_madry(loader, model, evaluate_size, epsilon, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    perrors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        if(i>=evaluate_size/y.shape[0]):
            break
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)


        # # perturb 
        _, pgd_err = _pgd(model, Variable(X), Variable(y), epsilon, 100, 0.0348)

        # print to logfile
        print(epoch, i, ce.data[0], err, file=log)

        # measure accuracy and record loss
        losses.update(ce.data[0], X.size(0))
        errors.update(err, X.size(0))
        perrors.update(pgd_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % verbose == 0: 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PGD Error {perror.val:.3f} ({perror.avg:.3f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors, perror=perrors))
        log.flush()

    print(' * PGD error {perror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(error=errors, perror=perrors))
    return errors.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

