import math, os, random, pickle, sys, csv, copy
import numpy as np
from tqdm import tqdm as tqdm

import torch.nn.functional as F
import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import dataset
import model
globalcoverage = []  # [{file, label, layercoverage, yhat}]
globalprofile = []
globallayercount = 0
currentlabels = []
convlayernumber = 7
for i in xrange(convlayernumber):
    globalprofile.append([])

gthreshold = 0.5 # neuron coverage threshold
def hook_all_conv_layer(net, handler):
    for l in net._modules.keys():
        if isinstance(net._modules.get(l), torch.nn.modules.conv.Conv2d):
            net._modules.get(l).register_forward_hook(handler)
        hook_all_conv_layer(net._modules.get(l), handler)


def get_channel_coverage_group_exp(self, input, output):
    from torchncoverage import NCoverage
    global globalcoverage
    nc = NCoverage(threshold=gthreshold)
    # print('Layer: ' + str(self))
    covered_channel_group = nc.get_channel_coverage_group(output.data)
    for c in xrange(len(covered_channel_group)):
        # print(c)

        d = -1 * (c + 1)

        if "layercoverage" not in globalcoverage[d]:
            globalcoverage[d]["layercoverage"] = []
        # total 7 cnn layer
        assert len(globalcoverage[d]["layercoverage"]) <= convlayernumber

        # print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))

        covered_channel = covered_channel_group[d]
        # print('total number of channels: ' + str(len(output.data[0])))
        # print('covered channels: ' + str(len(covered_channel)))
        # print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
        globalcoverage[d]["layercoverage"].append((len(output.data[0]), covered_channel))

    if len(globalcoverage[-1]["layercoverage"]) == convlayernumber:
        with open('globalcoveragecifar100_test_'+ str(gthreshold) + '.pickle', 'ab') as handle:
            pickle.dump(globalcoverage, handle, protocol=pickle.HIGHEST_PROTOCOL)
        globalcoverage = []

# collecting neuron coverage for all testing data
def get_test_coverage():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # data loader and model
    train_loader, test_loader = dataset.get100(batch_size=200, num_workers=1)
    cifar100_model = model.cifar100(128, pretrained=None)
    checkpoint = torch.load(os.path.join('./latest.pth'))
    cifar100_model.load_state_dict(checkpoint)
    cifar100_model.cuda()
    hook_all_conv_layer(cifar100_model, get_channel_coverage_group_exp)
    cifar100_model.eval()

    test_loss = 0
    correct = 0
    count = 0
    yhats = []
    labels = []
    print("collecting neuron coverage for all testing data...")

    t = tqdm(test_loader, desc="Evaluating on Test:")
    for batch_idx, (data, target) in enumerate(t):
        labels = labels + list(target)
        indx_target = target.clone()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        for i in xrange(len(target)):
            globalcoverage.append({})
            globalcoverage[-1]["dataset"] = "test"
        output = cifar100_model(data)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        yhats = yhats + list(pred.data.cpu().numpy())

        correct += pred.cpu().eq(indx_target).sum()
        count = count + len(target)
        if count % 1000 == 0:
            print("count: " + str(count))
    acc = 100. * correct / len(test_loader.dataset)

    print('acc: ' + str(acc))
    print(len(yhats))
    print(len(labels))
    with open('globalcifar100_test_yhats.pickle', 'wb') as handle:
        pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('globalcifar100_test_labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deepinspect(sample_10):
    #compute neuron-feature score
    #feature->neurons mapping
    total_layers = convlayernumber
    with open('globalcifar100_test_yhats.pickle', 'rb') as handle:
        test_yhats = pickle.load(handle)
    with open('globalcifar100_test_labels.pickle', 'rb') as handle:
        test_labels = pickle.load(handle)


    layer_coverage = [dict() for x in range(total_layers)]
    activation_sum = [dict() for x in range(total_layers)]
    features_sum = {}
    #compute features_sum    
    featuresmap = {}

    print("compute probability matrix... ")
    infile = open('globalcoveragecifar100_test_'+ str(gthreshold) + '.pickle', 'r')
    processed = 0
    while 1:
        try:
            globalcoverage = pickle.load(infile)
            assert len(globalcoverage) <= 200
            for g in globalcoverage:
                assert len(g["layercoverage"]) == convlayernumber
                if processed in sample_10:
                    processed = processed + 1
                    continue
               
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



    with open('p_layer_coverage_predicted_90_'+str(gthreshold)+'.pickle', 'wb') as handle:
        pickle.dump(p_layer_coverage, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('p_layer_coverage_predicted_90_'+str(gthreshold)+'.pickle', 'rb') as handle:
        p_layer_coverage = pickle.load(handle)



    labels_list = list(xrange(100))
    #compute pairwise distance
    #distance1 based on probability matrix
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
            distance1[(l1,l2)] = math.sqrt(d*1.0)/math.sqrt(normalization)

    distance = distance1 

    with open('neuron_distance_from_predicted_labels_test_90_'+str(gthreshold)+'.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["object","object", "neuron distance"])
        for key, value in sorted(distance.iteritems(), key=lambda (k,v): (v,k), reverse=False):

            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

def get_90_csv_files(sample_10):
    total_layers = convlayernumber
    with open('globalcifar100_test_yhats.pickle', 'rb') as handle:
        yhats_raw = pickle.load(handle)
    with open('globalcifar100_test_labels.pickle', 'rb') as handle:
        labels_raw = pickle.load(handle)



    labels_list = list(xrange(100))
    yhats = []
    labels = []
    imagefiles = []
    for i in xrange(len(yhats_raw)):
        if i not in sample_10:
            yhats.append(yhats_raw[i])
            labels.append(labels_raw[i])
            imagefiles.append(i)
            


    type1confusion = {}

    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in type1confusion):
                continue
            c = 0

            subcount = 0
            for i in xrange(len(yhats)):
                
                if l1 == labels[i]:
                    if l2 == yhats[i]:
                        c = c + 1
                    subcount = subcount + 1

            type1confusion[(l1,l2)] = c*1.0/subcount

    with open('objects_directional_type1_confusion_test_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type1 confusion"])
        
 
        for key, value in sorted(type1confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)


    with open('test_labels_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_file","labels"])
        
 
        for i in xrange(len(imagefiles)):
            csvrecord = []
            csvrecord.append(imagefiles[i])
            csvrecord.append(labels[i])
            writer.writerow(csvrecord)

    with open('test_predicted_labels_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_file","predicted labels"])
        
 
        for i in xrange(len(imagefiles)):
            csvrecord = []
            csvrecord.append(imagefiles[i])
            csvrecord.append(yhats[i])
            writer.writerow(csvrecord)

if __name__ == '__main__':
    get_test_coverage() # collecting neuron coverage for all testing data
    np.random.seed(0)
    sample_10 = np.random.choice(10000, 1000, replace=False) # keep randomly sampled 10% for other use
    deepinspect(sample_10=sample_10) # inspect the model using the model's prediction on 90% of testing data
    get_90_csv_files(sample_10=sample_10) # generate necessary information for 90% of testing data
    