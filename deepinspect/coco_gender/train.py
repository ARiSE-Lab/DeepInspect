import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm as tqdm

import torch.nn.functional as F
import torch, torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data_loader import CocoObject
from model import MultilabelObject

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./log/' ,
                        help='path for saving trained models and log info')
    parser.add_argument('--ann_dir', type=str, default='/home/yuchi/dataset/coco/annotations',
                        help='path for annotation json file')
    parser.add_argument('--image_dir', default = '/home/yuchi/dataset/coco/')

    parser.add_argument('--resume', default=0, type=int, help='whether to resume from log_dir if existent')
    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64) # batch size should be smaller if use text
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.log_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.log_dir))
        return   
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    #save all parameters for training
    with open(os.path.join(args.log_dir, "arguments.log"), "a") as f:
        f.write(str(args)+'\n')

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        normalize])
    val_transform = transforms.Compose([ 
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.crop_size), 
        transforms.ToTensor(), 
        normalize])

    # Data samplers.
    train_data = CocoObject(ann_dir = args.ann_dir, image_dir = args.image_dir, 
        split = 'train', transform = train_transform)

    val_data = CocoObject(ann_dir = args.ann_dir, image_dir = args.image_dir, 
        split = 'val', transform = val_transform)

    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, 
                                              shuffle = True, num_workers = 6,
                                              pin_memory = True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 4,
                                            pin_memory = True)

    # Build the models
    model = MultilabelObject(args, 81).cuda()
    criterion = nn.BCEWithLogitsLoss(weight = torch.FloatTensor(train_data.getObjectWeights()), size_average = True).cuda()

    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    optimizer = torch.optim.Adam(trainable_params(), args.learning_rate, weight_decay = 1e-5)

    best_performance = 0
    if args.resume:
        train_F = open(os.path.join(args.log_dir, 'train.csv'), 'a')
        val_F = open(os.path.join(args.log_dir, 'val.csv'), 'a')
        score_F = open(os.path.join(args.log_dir, 'score.csv'), 'a')
        if os.path.isfile(os.path.join(args.log_dir, 'checkpoint.pth.tar')):
            print("=> loading checkpoint '{}'".format(args.log_dir))
            checkpoint = torch.load(os.path.join(args.log_dir, 'checkpoint.pth.tar'))
            args.start_epoch = checkpoint['epoch']
            best_performance = checkpoint['best_performance']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))
    else:
        train_F = open(os.path.join(args.log_dir, 'train.csv'), 'w')
        val_F = open(os.path.join(args.log_dir, 'val.csv'), 'w')
        score_F = open(os.path.join(args.log_dir, 'score.csv'), 'w')


    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, epoch, model, criterion, train_loader, optimizer, train_F, score_F)
        current_performance = test(args, epoch, model, criterion, val_loader, optimizer, val_F, score_F)
        is_best = current_performance > best_performance
        best_performance = max(current_performance, best_performance)
        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance}
        save_checkpoint(args, model_state, is_best, os.path.join(args.log_dir, 'checkpoint.pth.tar'))
        os.system('python plot.py {} &'.format(args.log_dir))

    train_F.close()
    val_F.close()
    score_F.close()

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

    
def train(args, epoch, model, criterion, train_loader, optimizer, train_F, score_F):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_logger = AverageMeter()
    correct_logger = AverageMeter() 

    res = list()
    end = time.time()
    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, objects, image_ids) in enumerate(t):
        # if batch_idx == 100: break # constrain epoch size

        data_time.update(time.time() - end)
        # Set mini-batch dataset
        images = Variable(images).cuda()
        objects = Variable(objects).cuda()
        # Forward, Backward and Optimize
        optimizer.zero_grad()

        object_preds = model(images)
        loss = criterion(object_preds, objects)
        loss_logger.update(loss.data[0])

        object_preds_max = object_preds.data.max(1, keepdim=True)[1]
        object_correct = torch.gather(objects.data, 1, object_preds_max).cpu().sum()
        correct_logger.update(object_correct)

        res.append((image_ids, object_preds.data.cpu(), objects.data.cpu()))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        t.set_postfix(loss = loss_logger.avg)

        train_F.write('{},{},{}\n'.format(epoch, loss.data[0], object_correct))
        train_F.flush()

    # compute mean average precision score for object classifier
    preds_object   = torch.cat([entry[1] for entry in res], 0)
    targets_object = torch.cat([entry[2] for entry in res], 0)
    eval_score_object = average_precision_score(targets_object.numpy(), preds_object.numpy())
    print('\nmean average precision of object classifier on training data is {}\n'.format(eval_score_object))
    score_F.write('{},{},{}\n'.format(epoch, 'train', eval_score_object))
    score_F.flush()

def test(args, epoch, model, criterion, val_loader, optimizer, val_F, score_F):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_logger = AverageMeter()
    correct_logger = AverageMeter()

    res = list()
    end = time.time()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)
    for batch_idx, (images, objects, image_ids) in enumerate(t):
        # if batch_idx == 100: break # constrain epoch size

        data_time.update(time.time() - end)
        # Set mini-batch dataset
        images = Variable(images).cuda()
        objects = Variable(objects).cuda()

        object_preds = model(images)
        '''
        m = nn.Sigmoid()
        object_preds_r = m(object_preds)
        '''
        loss = criterion(object_preds, objects)
        loss_logger.update(loss.data[0])

        object_preds_max = object_preds.data.max(1, keepdim=True)[1]
        object_correct = torch.gather(objects.data, 1, object_preds_max).cpu().sum()
        correct_logger.update(object_correct)

        res.append((image_ids, object_preds.data.cpu(), objects.data.cpu()))

        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        t.set_postfix(loss = loss_logger.avg)

    val_F.write('{},{},{}\n'.format(epoch, loss_logger.avg, correct_logger.avg))
    val_F.flush()

    # compute mean average precision score for object classifier
    preds_object   = torch.cat([entry[1] for entry in res], 0)
    targets_object = torch.cat([entry[2] for entry in res], 0)
    eval_score_object = average_precision_score(targets_object.numpy(), preds_object.numpy())
    print('\nmean average precision of object classifier on val data is {}\n'.format(eval_score_object))
    score_F.write('{},{},{}\n'.format(epoch, 'test', eval_score_object))
    score_F.flush()

    return eval_score_object

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
                
if __name__ == '__main__':
    main()