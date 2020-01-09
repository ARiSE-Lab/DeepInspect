from __future__ import print_function
import os
import glob
import time
import numpy
import numpy as np
import csv
import math
import json
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import cv2
import cPickle as pickle
import random
from PIL import Image

globalcoverage = [] # [{file, label, layercoverage, yhat}]

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

class imagenet_inspect():


    def __init__(self):
        
        self.num_classes = 1000       
        self.IMG_SIZE = 224

        self.resnet_model = models.resnet50(pretrained=True).cuda()
        random.seed(0)
        #sample100 = random.sample(range(1000),100)
        
        with open('imagenet_class_index.json') as f:
            label_data = json.load(f)
        labeltoclass = {}
        for i in xrange(1000):
            labeltoclass[label_data[str(i)][0]] = i
        '''
        training_images = []
        training_labels = []

        for subdir, dirs, files in os.walk('/local/yuchi/train/'):
            for folder in dirs:
                if folder == "ILSVRC2012_img_train":
                    continue
                for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                    for file in folder_files:
                        #if labeltoclass[folder] in sample100:
                            training_images.append(os.path.join(folder_subdir, file))
                            training_labels.append(labeltoclass[folder])

        self.training_images = training_images
        self.training_labels = training_labels
        '''

        testing_images = []
        testing_images_unsampled = []
        testing_labels = []
        testing_labels_unsampled = []
        test_label_file = open("ILSVRC2012_validation_ground_truth.txt", "r")
        labels = test_label_file.readlines()
        mapping_file = open("ILSVRC2012_mapping.txt", "r")
        mapping = mapping_file.readlines()
        idtosynset = {}
        for m in mapping:
            temp = m.strip('\n').split(" ")
            idtosynset[int(temp[0])] = temp[1] 

        for l in labels:
            testing_labels_unsampled.append(labeltoclass[idtosynset[int(l)]])

        for subdir, dirs, files in os.walk('/local/yuchi/dataset/ILSVRC2012/val/'):
            print(len(files))
            for file in sorted(files):
                testing_images_unsampled.append(os.path.join(subdir, file))
                #print(file)
        for x,y in zip(testing_labels_unsampled, testing_images_unsampled):
            #if x in sample100:
            testing_labels.append(x)
            testing_images.append(y)
        self.testing_images = testing_images
        self.testing_labels = testing_labels

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])


        # define loss function (criterion) and pptimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.SGD(self.resnet_model.parameters(), 0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4)

    def test_pretrained_model(self):
        self.resnet_model.eval()
        batch_size = 128
        num_classes = 1000
       
        img = Image.open(self.testing_images[0]).convert('RGB')
        img = self.val_transform(img)
        

        img_batch = np.array([np.array(img)])
        img_batch = torch.from_numpy(img_batch)
        inputs = Variable(img_batch).cuda()
        output = self.resnet_model(inputs)
        print(output)
        output = output.cpu().data.numpy()
        print(np.argmax(output))
        print(self.testing_labels[0])

    def test_dnn_model(self, action = "test", modelid=0, weights_file="ILSVRC2012.hdf5"):
        
        batch_size = 50
        num_classes = 1000
        model = self.resnet_model
        model.eval()
        images = self.testing_images
        labels = self.testing_labels
        batch_num = math.floor(len(images) / batch_size)
        nice_n = math.floor(len(images) / batch_size) * batch_size

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        def get_batch():
            index = 0

            B = numpy.zeros(shape=(batch_size, 3, 224, 224))
            L = numpy.zeros(shape=(batch_size), dtype='int64')

            while index < batch_size:
                try:
                    img = Image.open(self.testing_images[self.current_index]).convert('RGB')
                    img = self.val_transform(img)
                    
                    B[index] = np.array(img)

                    L[index] = labels[self.current_index]

                    index = index + 1
                    self.current_index = self.current_index + 1
                except Exception, e:
                    print('error: '+ str(e))

                    print("Ignore image {}".format(images[self.current_index]))
                    self.current_index = self.current_index + 1

            return torch.from_numpy(B), torch.from_numpy(L)

        self.current_index = 0
        i = 0
        end = time.time()
        while self.current_index + batch_size <= len(images):
            b, l = get_batch()
            b = b.float()
            #l = l.float()
            print(self.current_index)
            print(len(images))
            b = Variable(b).cuda()
            l = Variable(l).cuda()
            output = model(b)
            #print(l.dtype)
            #print(output.dtype)
            
            output_debug = output.cpu().data.numpy()
            l_debug = l.cpu().data.numpy()
            print(np.argmax(output_debug, axis=1))
            print(l_debug)
            loss = self.criterion(output, l)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, l, topk=(1, 5))
            losses.update(loss.data[0], b.size(0))
            top1.update(prec1[0], b.size(0))
            top5.update(prec5[0], b.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if self.current_index % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, batch_num, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            i = i + 1
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    def get_coverage(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        batch_size = 50
        num_classes = 1000
        model = self.resnet_model
        #summary(model, (3, 224, 224))
        #exit()
        hook_all_conv_layer(model, get_channel_coverage_group_exp)
        model.eval()
        images = self.testing_images
        labels = self.testing_labels
        batch_num = math.floor(len(images) / batch_size)
        nice_n = math.floor(len(images) / batch_size) * batch_size

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        def get_batch():
            index = 0

            B = numpy.zeros(shape=(batch_size, 3, 224, 224))
            L = numpy.zeros(shape=(batch_size), dtype='int64')

            while index < batch_size:
                try:
                    img = Image.open(self.testing_images[self.current_index]).convert('RGB')
                    img = self.val_transform(img)
                    
                    B[index] = np.array(img)

                    L[index] = labels[self.current_index]

                    index = index + 1
                    self.current_index = self.current_index + 1
                except Exception, e:
                    print('error: '+ str(e))

                    print("Ignore image {}".format(images[self.current_index]))
                    self.current_index = self.current_index + 1

            return torch.from_numpy(B), torch.from_numpy(L)

        self.current_index = 0
        i = 0
        count = 0
        soft_yhats = []
        record_labels = []
        end = time.time()
        while self.current_index + batch_size <= len(images):
            b, l = get_batch()
            record_labels = record_labels + list(l)
            b = b.float()
            #l = l.float()
            print(self.current_index)
            print(len(images))
            b = Variable(b).cuda()
            l = Variable(l).cuda()
            
            for j in xrange(len(l)):
                globalcoverage.append({})
                globalcoverage[-1]["dataset"] = "test"

            output = model(b)
            #print(l.dtype)
            #print(output.dtype)
            
            #output_debug = output.cpu().data.numpy()
            #l_debug = l.cpu().data.numpy()
            #print(np.argmax(output_debug, axis=1))
            #print(l_debug)
            loss = self.criterion(output, l)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, l, topk=(1, 5))
            losses.update(loss.data[0], b.size(0))
            top1.update(prec1[0], b.size(0))
            top5.update(prec5[0], b.size(0))

            m = torch.nn.Sigmoid()
            output1 = m(output)
            soft_yhats = soft_yhats + list(output1.data.cpu().numpy())
            count = count + len(l)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if self.current_index % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, batch_num, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            i = i + 1
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
        print(i)
        print(len(soft_yhats))
        print(len(record_labels))
        with open('globalimagenet_test_sigmoid_yhats.pickle', 'wb') as handle:
            pickle.dump(soft_yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('globalimagenet_test_labels.pickle', 'wb') as handle:
            pickle.dump(record_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_labels(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        batch_size = 50
        num_classes = 1000
        model = self.resnet_model
        #summary(model, (3, 224, 224))
        #exit()
        #hook_all_conv_layer(model, get_channel_coverage_group_exp)
        model.eval()
        images = self.testing_images
        labels = self.testing_labels
        batch_num = math.floor(len(images) / batch_size)
        nice_n = math.floor(len(images) / batch_size) * batch_size

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        def get_batch():
            index = 0

            B = numpy.zeros(shape=(batch_size, 3, 224, 224))
            L = numpy.zeros(shape=(batch_size), dtype='int64')

            while index < batch_size:
                try:
                    img = Image.open(self.testing_images[self.current_index]).convert('RGB')
                    img = self.val_transform(img)
                    
                    B[index] = np.array(img)

                    L[index] = labels[self.current_index]

                    index = index + 1
                    self.current_index = self.current_index + 1
                except Exception, e:
                    print('error: '+ str(e))

                    print("Ignore image {}".format(images[self.current_index]))
                    self.current_index = self.current_index + 1

            return torch.from_numpy(B), torch.from_numpy(L)

        self.current_index = 0
        i = 0
        count = 0
        soft_yhats = []
        record_labels = []
        end = time.time()
        while self.current_index + batch_size <= len(images):
            b, l = get_batch()
            record_labels = record_labels + list(l.data.cpu().numpy())
            count = count + len(l)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if self.current_index % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, batch_num, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            i = i + 1
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
        print(i)
        #print(len(soft_yhats))
        print(record_labels[0])
        print(len(record_labels))
        with open('globalimagenet_test_labels.pickle', 'wb') as handle:
            pickle.dump(record_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def verify_sigmoid_labels(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        batch_size = 50
        num_classes = 1000
        model = self.resnet_model
        #summary(model, (3, 224, 224))
        #exit()

        model.eval()
        images = self.testing_images
        labels = self.testing_labels
        batch_num = math.floor(len(images) / batch_size)
        nice_n = math.floor(len(images) / batch_size) * batch_size

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        def get_batch():
            index = 0

            B = numpy.zeros(shape=(batch_size, 3, 224, 224))
            L = numpy.zeros(shape=(batch_size), dtype='int64')

            while index < batch_size:
                try:
                    img = Image.open(self.testing_images[self.current_index]).convert('RGB')
                    img = self.val_transform(img)
                    
                    B[index] = np.array(img)

                    L[index] = labels[self.current_index]

                    index = index + 1
                    self.current_index = self.current_index + 1
                except Exception, e:
                    print('error: '+ str(e))

                    print("Ignore image {}".format(images[self.current_index]))
                    self.current_index = self.current_index + 1

            return torch.from_numpy(B), torch.from_numpy(L)

        self.current_index = 0
        i = 0
        count = 0
        soft_yhats = []
        yhats = []
        record_labels = []
        end = time.time()
        while self.current_index + batch_size <= len(images):
            b, l = get_batch()
            record_labels = record_labels + list(l)
            b = b.float()
            #l = l.float()
            print(self.current_index)
            print(len(images))
            b = Variable(b).cuda()
            l = Variable(l).cuda()
            
          
            output = model(b)
            #print(l.dtype)
            #print(output.dtype)
            
            #output_debug = output.cpu().data.numpy()
            #l_debug = l.cpu().data.numpy()
            #print(np.argmax(output_debug, axis=1))
            #print(l_debug)
            loss = self.criterion(output, l)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, l, topk=(1, 5))
            losses.update(loss.data[0], b.size(0))
            top1.update(prec1[0], b.size(0))
            top5.update(prec5[0], b.size(0))

            m = torch.nn.Sigmoid()
            output1 = m(output)
            soft_yhats = soft_yhats + list(output1.data.cpu().numpy())
            yhats = yhats + list(output.data.cpu().numpy())
            count = count + len(l)

            y = np.argmax(output.data.cpu().numpy(), axis=1)
            y1 = np.argmax(output1.data.cpu().numpy(), axis=1)
            print(y)
            print(y1)
            if (y == y1).all():
                print ("verified")
            else:
                print ("incorrect")
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if self.current_index % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, batch_num, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            i = i + 1
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
        print(i)
        print(len(soft_yhats))
        print(len(record_labels))
        with open('verify_test_sigmoid_yhats.pickle', 'wb') as handle:
            pickle.dump(soft_yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('verify_test_yhats.pickle', 'wb') as handle:
            pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def hook_all_conv_layer(net, handler):
    for l in net._modules.keys():
        if isinstance(net._modules.get(l), torch.nn.modules.conv.Conv2d):
            net._modules.get(l).register_forward_hook(handler)
        hook_all_conv_layer(net._modules.get(l), handler)

def get_channel_coverage_group_exp(self, input, output):
    from torchncoverage import NCoverage
    global globalcoverage
    nc = NCoverage(threshold = 0.5)
    #print('Layer: ' + str(self))
    covered_channel_group = nc.get_channel_coverage_group(output.data)
    for c in xrange(len(covered_channel_group)):
        #print(c)

        d = -1*(c+1)

        if "layercoverage" not in globalcoverage[d]:
            globalcoverage[d]["layercoverage"] = []
        # total 7 cnn layer
        assert len(globalcoverage[d]["layercoverage"]) <= 53

            
        #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
        
        covered_channel = covered_channel_group[d]
        #print('total number of channels: ' + str(len(output.data[0])))
        #print('covered channels: ' + str(len(covered_channel)))
        #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
        globalcoverage[d]["layercoverage"].append((len(output.data[0]), covered_channel))

    if len(globalcoverage[-1]["layercoverage"]) == 53:
        #with open('globalcoveragecocoexp_val.pickle', 'ab') as handle:
        with open('globalcoverageimagenet_test.pickle', 'ab') as handle:
            pickle.dump(globalcoverage, handle, protocol=pickle.HIGHEST_PROTOCOL)
        globalcoverage = []

if __name__ == '__main__':
    md = imagenet_inspect()
    md.get_labels()
    #md.test_pretrained_model()
    #md.test_dnn_model()
    md.verify_sigmoid_labels()
    md.get_coverage()
    

    
