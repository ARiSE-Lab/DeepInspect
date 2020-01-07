import argparse, os, string, pdb, time, copy, math, random
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import pickle
import numpy as np
import csv
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
import problems as pblm
from trainer import *


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(0)
np.random.seed(0)
gthreshold = 0.5 # neuron coverage threshold
globalcoverage = [] # [{file, label, layercoverage, yhat}]
hook_layer_count = 0

train_loader, test_loader = pblm.cifar_loaders(64)

def select_model(m): 
    if m == 'large': 
        # raise ValueError
        model = pblm.cifar_model_large().cuda()
    elif m == 'resnet': 
        model = pblm.cifar_model_resnet(N=args.resnet_N, factor=args.resnet_factor).cuda()
    else: 
        model = pblm.cifar_model().cuda() 
    summary(model, (3, 32, 32))
    return model

def get_yhats_test():
    model = select_model('resnet')
    model_name = 'model/cifar_resnet_robust_new.h5'
    model = torch.load(model_name)[0].cuda()
    model.eval()

    t = tqdm(test_loader, desc="Evaluating on Test:")
    yhats = []
    labels = []
    for batch_idx, (data, target) in enumerate(t):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target) 

        object_preds = model(data)     
        for i in xrange(len(data)):
            yhat = []
            label = []
            yhat.append(np.argmax(object_preds[i].cpu().data.numpy()))
            label.append(int(target[i].cpu().data.numpy()))
            yhats.append(yhat)
            labels.append(label)
            #print(label)

    print(labels[0])
    print(yhats[:10])
    print("total sampled test data")
    print(len(labels))
    print(len(yhats))
    with open('robust_cifar_globalyhats_test.pickle', 'wb') as handle:
        pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('robust_cifar_globallabels_test.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def hook_all_conv_layer(net, handler):
    global hook_layer_count
    for l in net._modules.keys():
        if isinstance(net._modules.get(l), torch.nn.modules.conv.Conv2d):
            net._modules.get(l).register_forward_hook(handler)
            hook_layer_count = hook_layer_count + 1
        hook_all_conv_layer(net._modules.get(l), handler)

def hook_all_layer(net, handler):
    global hook_layer_count
    for l in net._modules.keys():
        if isinstance(net._modules.get(l), torch.nn.modules.Linear) or isinstance(net._modules.get(l), torch.nn.modules.conv.Conv2d):
            net._modules.get(l).register_forward_hook(handler)
            hook_layer_count = hook_layer_count + 1
        hook_all_layer(net._modules.get(l), handler)

def get_channel_coverage_group_exp(self, input, output):
    from torchncoverage import NCoverage
    global globalcoverage
    nc = NCoverage(threshold=gthreshold)
    #print('Layer: ' + str(self))
    #print(len(output.data)) 64
    covered_channel_group = nc.get_channel_coverage_group(output.data)
    for c in xrange(len(covered_channel_group)):
        #print(c)

        d = -1*(c+1)

        if "layercoverage" not in globalcoverage[d]:
            globalcoverage[d]["layercoverage"] = []

        assert len(globalcoverage[d]["layercoverage"]) <= 7 # number of conv layer

            
        #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
        
        covered_channel = covered_channel_group[d]
        #print('total number of channels: ' + str(len(output.data[0])))
        #print('covered channels: ' + str(len(covered_channel)))
        #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
        globalcoverage[d]["layercoverage"].append((len(output.data[0]), covered_channel))

    if len(globalcoverage[-1]["layercoverage"]) == 7:
        with open('globalcoveragecifarexp_small_test_'+str(gthreshold)+'.pickle', 'ab') as handle:
            pickle.dump(globalcoverage, handle, protocol=pickle.HIGHEST_PROTOCOL)
        globalcoverage = []

#Get coverage of all validating data in train dataset.
def get_coverage_test():
    global globalcoverage
    global hook_layer_count
    hook_layer_count = 0
    model = select_model('resnet')
    model_name = 'model/cifar_resnet_robust_new.h5'
    model = torch.load(model_name)[0].cuda()
    

    
    hook_all_layer(model, get_channel_coverage_group_exp)
    print("total hook layer: " + str(hook_layer_count))
    hook_layer_count = 0
    #exit()
    model.eval()
    
    count = 0
    yhats = []
    labels = []

    t = tqdm(test_loader, desc="Evaluating on Test:")
    s_function = nn.Sigmoid()
    for batch_idx, (data, target) in enumerate(t):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target) 

        for i in xrange(len(data)):
            globalcoverage.append({})
            yhat = []
            label = []

            label.append(int(target[i].cpu().data.numpy()))
            globalcoverage[-1]["file"] = str(count)
            globalcoverage[-1]["yhat"] = yhat
            globalcoverage[-1]["dataset"] = "test"
            globalcoverage[-1]["jlabel"] = label
            count = count + 1

        object_preds = model(data)

def get_10_csv_files(sample_10):

    with open('robust_cifar_globalyhats_test.pickle', 'rb') as handle:
        yhats_raw = pickle.load(handle)
    with open('robust_cifar_globallabels_test.pickle', 'rb') as handle:
        labels_raw = pickle.load(handle)

    yhats = []
    labels = []

    for i in xrange(len(yhats_raw)):
        if i in sample_10:
            yhats.append(yhats_raw[i])
            labels.append(labels_raw[i])

    with open('robust_cifar_resnet_test_labels_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","labels"])
        
 
        for i in xrange(len(labels)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(";".join(map(str,labels[i])))
            writer.writerow(csvrecord)

    with open('robust_cifar_resnet_test_predicted_labels_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","predicted labels"])
        
 
        for i in xrange(len(yhats)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(";".join(map(str,yhats[i])))
            writer.writerow(csvrecord)

    total_layers = 15


    labels_list = []
    for i in xrange(10):
        labels_list.append(i)


    type1confusion = {}
    type2confusion = {}

    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in type2confusion):
                continue
            c4 = 0
            c5 = 0
            c6 = 0


            #confusion4,5,6
            #'confusion((0;1)->(1;0) or (1;0)->(0;1))', 'confusion((0;1)or(1;0)->(1;1))','confusion((0;1)or(1;0)->(0;0))'
            subcount = 0
            for i in xrange(len(yhats)):
                
                if l1 in labels[i] and l2 not in labels[i]:
                    #confusion4
                    if l2 in yhats[i] and l1 not in yhats[i]:
                        c4 = c4 + 1
                        
                    #confusion5
                    elif l1 in yhats[i] and l2 in yhats[i]:
                        c5 = c5 + 1
                    #confusion6
                    elif l1 not in yhats[i] and l2 not in yhats[i]:
                        c6 = c6 + 1
                    subcount = subcount + 1

            
            if subcount < 10:
                continue

            type2confusion[(l1,l2)] = c5*1.0/subcount
            type1confusion[(l1,l2)] = c4*1.0/subcount
    distance = type1confusion


    with open('robust_cifar_resnet_objects_directional_type1_confusion_test_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type1 confusion"])
        
 
        for key, value in sorted(type1confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

def get_90_csv_files(sample_10):

    with open('robust_cifar_globalyhats_test.pickle', 'rb') as handle:
        yhats_raw = pickle.load(handle)
    with open('robust_cifar_globallabels_test.pickle', 'rb') as handle:
        labels_raw = pickle.load(handle)

    yhats = []
    labels = []

    for i in xrange(len(yhats_raw)):
        if i not in sample_10:
            yhats.append(yhats_raw[i])
            labels.append(labels_raw[i])

    with open('robust_cifar_resnet_test_labels_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","labels"])
        
 
        for i in xrange(len(labels)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(";".join(map(str,labels[i])))
            writer.writerow(csvrecord)

    with open('robust_cifar_resnet_test_predicted_labels_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","predicted labels"])
        
 
        for i in xrange(len(yhats)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(";".join(map(str,yhats[i])))
            writer.writerow(csvrecord)

    total_layers = 15


    labels_list = []
    for i in xrange(10):
        labels_list.append(i)


    type1confusion = {}
    type2confusion = {}

    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in type2confusion):
                continue
            c4 = 0
            c5 = 0
            c6 = 0


            #confusion4,5,6
            #'confusion((0;1)->(1;0) or (1;0)->(0;1))', 'confusion((0;1)or(1;0)->(1;1))','confusion((0;1)or(1;0)->(0;0))'
            subcount = 0
            for i in xrange(len(yhats)):
                
                if l1 in labels[i] and l2 not in labels[i]:
                    #confusion4
                    if l2 in yhats[i] and l1 not in yhats[i]:
                        c4 = c4 + 1
                    subcount = subcount + 1
            if subcount < 10:
                continue

            type1confusion[(l1,l2)] = c4*1.0/subcount
    distance = type1confusion


    with open('robust_cifar_resnet_objects_directional_type1_confusion_test_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type1 confusion"])
        
 
        for key, value in sorted(type1confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

def deepinspect(sample_10):
    #compute neuron-feature score
    #feature->neurons mapping
    total_layers = 15
    with open('robust_cifar_globalyhats_test.pickle', 'rb') as handle:
        test_yhats = pickle.load(handle)
    with open('robust_cifar_globallabels_test.pickle', 'rb') as handle:
        test_labels = pickle.load(handle)
    layer_coverage = [dict() for x in range(total_layers)]
    activation_sum = [dict() for x in range(total_layers)]
    features_sum = {}
    #compute features_sum    
    featuresmap = {}

    infile = open('globalcoveragecifarexp_resnet_test_'+str(gthreshold)+'.pickle', 'r')
    processed = 0
    while 1:
        try:
            globalcoverage = pickle.load(infile)
            assert len(globalcoverage) <= 64
            for g in globalcoverage:
                assert len(g["layercoverage"]) == total_layers
                if processed in sample_10:
                    processed = processed + 1
                    continue
                from sets import Set
                
                 
                #for v in list(set(g["jlabel"])):
                #print(test_yhats[processed])
                for v in list(set(test_yhats[processed])):
                    if v not in features_sum.keys():
                        features_sum[v] = 1
                    else:
                        features_sum[v] = features_sum[v] + 1
                
                
                #compute features scores                
                for l in xrange(total_layers):            
                    for j in g["layercoverage"][l][1]:

                        #compute activation_sum
                        if j not in activation_sum[l].keys():
                            activation_sum[l][j] = 1
                        else:
                            activation_sum[l][j] = activation_sum[l][j] + 1

                        if j not in layer_coverage[l].keys():
                            layer_coverage[l][j] = {}
                        #for v in list(set(g["jlabel"])):
                        for v in list(set(test_yhats[processed])):
                            if v not in layer_coverage[l][j].keys():
                                layer_coverage[l][j][v] = 1
                            else:
                                layer_coverage[l][j][v] = layer_coverage[l][j][v] + 1

                processed = processed + 1

            if processed%1024 == 0:
                print(processed)
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()

    with open('labels_count_test.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['labels', 'count'])
        for key, value in sorted(features_sum.iteritems(), key=lambda (k,v): (v,k), reverse=True):
            csvrecord = []
            csvrecord.append(key)
            csvrecord.append(value)
            writer.writerow(csvrecord)

    #nomalize score(feature probability with respect to each neuron)
    p_layer_coverage = copy.deepcopy(layer_coverage)

    for l in xrange(total_layers):
        for n in layer_coverage[l].keys():
            for v in layer_coverage[l][n].keys():
                p_layer_coverage[l][n][v] = layer_coverage[l][n][v]*1.0 / features_sum[v]

    with open('cifar_p_layer_coverage_predicted_90_'+str(gthreshold)+'.pickle', 'wb') as handle:
        pickle.dump(p_layer_coverage, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cifar_p_layer_coverage_predicted_90_'+str(gthreshold)+'.pickle', 'rb') as handle:
        p_layer_coverage = pickle.load(handle)

 

    labels_list = []
    for i in xrange(10):
        labels_list.append(i)


    distance1 = {}
    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in distance1) or ((l2, l1) in distance1):
                continue
            d = 0;
            normalization = 0
            for l in xrange(total_layers):
                for n in p_layer_coverage[l].keys():
                    if l1 in p_layer_coverage[l][n] and l2 in p_layer_coverage[l][n]:
                        d = d + (p_layer_coverage[l][n][l1] - p_layer_coverage[l][n][l2])**2
                        normalization = normalization + 1
                    elif l1 in p_layer_coverage[l][n] and l2 not in p_layer_coverage[l][n]:
                        d = d + p_layer_coverage[l][n][l1]**2
                        normalization = normalization + 1
                    elif l2 in p_layer_coverage[l][n] and l1 not in p_layer_coverage[l][n]:
                        d = d + p_layer_coverage[l][n][l2]**2
                        normalization = normalization + 1
            if normalization == 0:
                continue
            distance1[(l1,l2)] = math.sqrt(d*1.0)/math.sqrt(normalization)
    print("len of distance")
    print(len(distance1))

    with open('cifar_distance.pickle', 'wb') as handle:
        pickle.dump(distance1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    distance = distance1

    with open('robust_cifar_resnet_neuron_distance_from_predicted_labels_test_90_'+str(gthreshold)+'.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["object","object", "neuron distance"])
        for key, value in sorted(distance.iteritems(), key=lambda (k,v): (v,k), reverse=False):

            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)


if __name__ == '__main__':

    #Running original model in test on cifar10 test data to see if the model works.
    #Generate predicted labels and ground truth labels in pickle.
    get_yhats_test()
    #Get neuron coverage
    get_coverage_test()


    np.random.seed(0)
    sample_10 = np.random.choice(10000, 1000, replace=False)
    # Get object pair-wise probability distance matrix
    deepinspect(sample_10=sample_10)
    #Compute pair-wise type1 confusion errors and write pairwise confusion errors and predicted labels into files.
    #get_10_csv_files(sample_10=sample_10)
    get_90_csv_files(sample_10=sample_10)
