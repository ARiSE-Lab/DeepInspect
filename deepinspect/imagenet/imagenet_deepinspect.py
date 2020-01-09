import math, os, random, json, sys, pdb, csv, copy
import string, shutil, time, argparse
import numpy as np
import cPickle as pickle
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

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
random.seed(0)
labels_list = list(range(1000))

def verify_accuracy():
    with open('verify_test_yhats.pickle', 'rb') as handle:
        test_yhats = pickle.load(handle)
    with open('globalimagenet_test_labels.pickle', 'rb') as handle:
        labels = pickle.load(handle)
    yhats = np.argmax(test_yhats, axis=1)
    assert len(yhats) == len(labels)
    count = 0
    for i in xrange(50000):
        if yhats[i] == labels[i]:
            count = count + 1
    print(count*1.0/50000)


def deepinspect(sample_10):
    #compute neuron-feature score
    #feature->neurons mapping
    with open('verify_test_yhats.pickle', 'rb') as handle:
        test_yhats = pickle.load(handle)
    with open('globalimagenet_test_labels.pickle', 'rb') as handle:
        labels = pickle.load(handle)
    yhats = np.argmax(test_yhats, axis=1)

    with open('test_yhats.pickle', 'wb') as handle:
        pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    total_layers = 53
    with open('test_yhats.pickle', 'rb') as handle:
        test_yhats = pickle.load(handle)
    with open('globalimagenet_test_labels.pickle', 'rb') as handle:
        test_labels = pickle.load(handle)


    print(len(test_yhats))
    layer_coverage = [dict() for x in range(total_layers)]
    activation_sum = [dict() for x in range(total_layers)]
    features_sum = {}
    #compute features_sum    
    featuresmap = {}

    infile = open('globalcoverageimagenet_test.pickle', 'r')
    processed = 0
    total_processed = 0
    while 1:
        try:
            globalcoverage = pickle.load(infile)
            assert len(globalcoverage) == 50
            for g in globalcoverage:
                if test_labels[processed] not in labels_list:
                    processed = processed + 1
                    continue
                if processed in sample_10:
                    processed = processed + 1
                    continue
                assert len(g["layercoverage"]) == 53
                
                total_processed = total_processed + 1
                from sets import Set
                
                 
                #for v in list(set(g["jlabel"])):
                for v in list(set([test_yhats[processed]])):
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
                        for v in list(set([test_yhats[processed]])):
                            if v not in layer_coverage[l][j].keys():
                                layer_coverage[l][j][v] = 1
                            else:
                                layer_coverage[l][j][v] = layer_coverage[l][j][v] + 1

                processed = processed + 1

            if processed%500 == 0:
                print("processed data: " + str(processed))
                print("processed sampled data: " + str(total_processed))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()
    print("total processed sampled data: " + str(total_processed))
    with open('predicted_labels_count_test.csv', 'wb',0) as csvfile1:
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

    with open('p_layer_coverage_predicted_90.pickle', 'wb') as handle:
        pickle.dump(p_layer_coverage, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_10_csv_files(sample_10):
    import numpy as np

    total_layers = 53
    print(len(labels_list))
    with open('test_yhats.pickle', 'rb') as handle:
        yhats_raw = pickle.load(handle)
    with open('globalimagenet_test_labels.pickle', 'rb') as handle:
        labels_raw = pickle.load(handle)
    #with open('p_feature_sensitivity_predicted_sigmoid.pickle', 'rb') as handle:
        #p_feature_sensitivity = pickle.load(handle)
    #with open('p_layer_coverage_predicted_90.pickle', 'rb') as handle:
        #p_layer_coverage = pickle.load(handle)

    yhats = []
    labels = []
    for i in xrange(len(yhats_raw)):
        if i in sample_10:
            yhats.append(yhats_raw[i])
            labels.append(labels_raw[i])


    x4 = []
    x5 = [] 
    y4 = []
    y5 = [] 
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    truth = {}
    confusion = {}
    label_object_count = {}
    for i in labels_list:
        label_object_count[i] = 0
    for i in xrange(len(labels)):
        if labels[i] in labels_list:
            label_object_count[int(labels[i])] = label_object_count[int(labels[i])] + 1

    confusion_pair_count = {}
    #print(labels)
    for i in xrange(len(yhats)):
        if labels[i] in labels_list and yhats[i] in labels_list:
            #print(labels[i])
            if int(labels[i]) != int(yhats[i]):
                if (int(labels[i]),int(yhats[i])) not in confusion_pair_count:
                    confusion_pair_count[(int(labels[i]),int(yhats[i]))] = 1
                else:
                    confusion_pair_count[(int(labels[i]),int(yhats[i]))] = confusion_pair_count[(int(labels[i]),int(yhats[i]))] + 1

    directional_confusion = {}
    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in directional_confusion):
                continue

            if (l1,l2) not in confusion_pair_count:
                directional_confusion[(l1,l2)] = 0
            else:
                directional_confusion[(l1,l2)] = confusion_pair_count[(l1,l2)]*1.0/label_object_count[l1]
    
    with open('objects_directional_type1_confusion_test_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type1 confusion"])
        
 
        for key, value in sorted(directional_confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

    with open('test_labels_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","labels"])
        
 
        for i in xrange(len(labels)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(labels[i])
            writer.writerow(csvrecord)

    with open('test_predicted_labels_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","predicted labels"])
        
 
        for i in xrange(len(yhats)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(yhats[i])
            writer.writerow(csvrecord)

def get_90_csv_files(sample_10):
    import numpy as np

    total_layers = 53
    print(len(labels_list))
    with open('test_yhats.pickle', 'rb') as handle:
        yhats_raw = pickle.load(handle)
    with open('globalimagenet_test_labels.pickle', 'rb') as handle:
        labels_raw = pickle.load(handle)
    #with open('p_feature_sensitivity_predicted_sigmoid.pickle', 'rb') as handle:
        #p_feature_sensitivity = pickle.load(handle)
    #with open('p_layer_coverage_predicted_90.pickle', 'rb') as handle:
        #p_layer_coverage = pickle.load(handle)

    yhats = []
    labels = []
    for i in xrange(len(yhats_raw)):
        if i not in sample_10:
            yhats.append(yhats_raw[i])
            labels.append(labels_raw[i])

    x4 = []
    x5 = [] 
    y4 = []
    y5 = [] 
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    truth = {}
    confusion = {}
    label_object_count = {}
    for i in labels_list:
        label_object_count[i] = 0
    for i in xrange(len(labels)):
        if labels[i] in labels_list:
            label_object_count[int(labels[i])] = label_object_count[int(labels[i])] + 1

    confusion_pair_count = {}
    #print(labels)
    for i in xrange(len(yhats)):
        if labels[i] in labels_list and yhats[i] in labels_list:
            #print(labels[i])
            if int(labels[i]) != int(yhats[i]):
                if (int(labels[i]),int(yhats[i])) not in confusion_pair_count:
                    confusion_pair_count[(int(labels[i]),int(yhats[i]))] = 1
                else:
                    confusion_pair_count[(int(labels[i]),int(yhats[i]))] = confusion_pair_count[(int(labels[i]),int(yhats[i]))] + 1

    directional_confusion = {}
    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in directional_confusion):
                continue

            if (l1,l2) not in confusion_pair_count:
                directional_confusion[(l1,l2)] = 0
            else:
                directional_confusion[(l1,l2)] = confusion_pair_count[(l1,l2)]*1.0/label_object_count[l1]
    
    with open('objects_directional_type1_confusion_test_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type1 confusion"])
        
 
        for key, value in sorted(directional_confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

    with open('test_labels_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","labels"])
        
 
        for i in xrange(len(labels)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(labels[i])
            writer.writerow(csvrecord)

    with open('test_predicted_labels_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_id","predicted labels"])
        
 
        for i in xrange(len(yhats)):
            csvrecord = []
            csvrecord.append(str(i))
            csvrecord.append(yhats[i])
            writer.writerow(csvrecord)

if __name__ == '__main__':

    verify_accuracy()
    #29600
    np.random.seed(0)
    sample_10 = np.random.choice(50000, 5000, replace=False)
    
    get_10_csv_files(sample_10=sample_10)
    get_90_csv_files(sample_10=sample_10)
    read_coverage_dump_probability_test_predicted_90(sample_10=sample_10)
 
    