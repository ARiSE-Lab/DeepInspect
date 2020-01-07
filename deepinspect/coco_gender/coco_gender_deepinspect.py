import math, os, random, json, pickle, sys, pdb, csv, copy
import string, shutil, time, argparse
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm as tqdm
import matplotlib
matplotlib.use('Agg')
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

globalcoverage = [] # [{file, label, layercoverage, yhat}]

def get_id2object():
    ann_dir = '/home/yuchi/dataset/coco/annotations'
    image_dir = '/home/yuchi/dataset/coco/'    
    
    from pycocotools.coco import COCO

    ann_path = os.path.join(ann_dir, "instances_train2014.json")
    cocoAPI = COCO(ann_path)
    data = json.load(open(ann_path))
    #81 objects

    id2object = dict()
    object2id = dict()
    person_id = -1
    for idx, elem in enumerate(data['categories']):
        if elem['name'] == 'person':
            person_id = idx
            print("person index: " + str(idx))
            id2object[idx] = "man"
            object2id["man"] = idx
            continue
        id2object[idx] = elem['name']
        object2id[elem['name']] = idx
    id2object[80] = "woman"
    object2id['woman'] = 80
    assert person_id != -1
    return id2object

def hook_all_conv_layer(net, handler):
    for l in net._modules.keys():
        if isinstance(net._modules.get(l), torch.nn.modules.conv.Conv2d):
            net._modules.get(l).register_forward_hook(handler)
        hook_all_conv_layer(net._modules.get(l), handler)

def get_channel_coverage_group_exp(self, input, output):
    from torchncoverage import NCoverage
    global globalcoverage
    nc = NCoverage(threshold = 0.25)
    #print('Layer: ' + str(self))
    covered_channel_group = nc.get_channel_coverage_group(output.data)
    for c in xrange(len(covered_channel_group)):
        #print(c)

        d = -1*(c+1)

        if "layercoverage" not in globalcoverage[d]:
            globalcoverage[d]["layercoverage"] = []
        # total 53 cnn layer
        assert len(globalcoverage[d]["layercoverage"]) <= 53

            
        #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
        
        covered_channel = covered_channel_group[d]
        #print('total number of channels: ' + str(len(output.data[0])))
        #print('covered channels: ' + str(len(covered_channel)))
        #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
        globalcoverage[d]["layercoverage"].append((len(output.data[0]), covered_channel))

    if len(globalcoverage[-1]["layercoverage"]) == 53:
        with open('globalcoveragecocoexp_test_0.25.pickle', 'ab') as handle:
            pickle.dump(globalcoverage, handle, protocol=pickle.HIGHEST_PROTOCOL)
        globalcoverage = []
    #print(len(globalcoverage[-1]["layercoverage"]))

def get_id2object_pkl():
    ann_dir = '/home/yuchi/dataset/coco/annotations'
    image_dir = '/home/yuchi/dataset/coco/'    
    
    from pycocotools.coco import COCO

    ann_path = os.path.join(ann_dir, "instances_train2014.json")
    cocoAPI = COCO(ann_path)
    data = json.load(open(ann_path))
    #81 objects

    id2object = dict()
    object2id = dict()
    person_id = -1
    for idx, elem in enumerate(data['categories']):
        if elem['name'] == 'person':
            person_id = idx
            print("person index: " + str(idx))
            id2object[idx] = "man"
            object2id["man"] = idx
            continue
        id2object[idx] = elem['name']
        object2id[elem['name']] = idx
    id2object[80] = "woman"
    object2id['woman'] = 80
    assert person_id != -1
    with open('id2object.pickle', 'wb') as handle:
        pickle.dump(id2object, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Get coverage of all validating data in train dataset.
def get_coverage_test():
    global globalcoverage
    ann_dir = '/local/yuchi/dataset/coco/annotations'
    image_dir = '/local/yuchi/dataset/coco/'
    crop_size = 224
    image_size = 256
    batch_size = 16
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])

    val_transform = transforms.Compose([ 
        transforms.Scale(image_size),
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(), 
        normalize])

    # Data samplers.
    train_data = CocoObject(ann_dir = ann_dir, image_dir = image_dir, 
        split = 'test', transform = val_transform)
    image_ids = train_data.new_image_ids 
    image_path_map = train_data.image_path_map
    #80 objects
    id2object = train_data.id2object
    id2labels = train_data.id2labels
    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                            shuffle = False, num_workers = 4,
                                            pin_memory = True)
    model = MultilabelObject(None, 81).cuda()
    hook_all_conv_layer(model, get_channel_coverage_group_exp)
    log_dir = "./log/"
    log_dir1 =  "/home/yuchi/work/coco/backup"
    checkpoint = torch.load(os.path.join(log_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    t = tqdm(train_loader, desc = 'Activation')
    count = 0
    for batch_idx, (images, objects, image_ids) in enumerate(t):

        images = Variable(images).cuda()
        objects = Variable(objects).cuda()

        for i in xrange(len(image_ids)):
            globalcoverage.append({})
            image_file_name = image_path_map[int(image_ids[i])]
            yhat = []
            '''
            for j in xrange(len(object_preds[i])):
                a = object_preds_r[i][j].cpu().data.numpy()
                if a[0] > 0.5:
                    yhat.append(id2object[j])
            '''
            globalcoverage[-1]["file"] = image_file_name
            globalcoverage[-1]["yhat"] = yhat
            globalcoverage[-1]["dataset"] = "test"
            globalcoverage[-1]["jlabel"] = id2labels[int(image_ids[i])]
        
        object_preds = model(images) 
        m = nn.Sigmoid()
        object_preds_r = m(object_preds)
        count = count + len(image_ids)
        

        if count % 1000 == 0:
            print("count: " + str(count))
    #import pickle
    #with open('globalcoveragecocoexp_val.pickle', 'ab') as handle:
        #pickle.dump(globalcoverage, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Get yhats
def get_yhats_val():

    
    ann_dir = '/home/yuchi/dataset/coco/annotations'
    image_dir = '/home/yuchi/dataset/coco/'
    crop_size = 224
    image_size = 256
    batch_size = 16
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])

    val_transform = transforms.Compose([ 
        transforms.Scale(image_size),
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(), 
        normalize])

    # Data samplers.
    train_data = CocoObject(ann_dir = ann_dir, image_dir = image_dir, 
        split = 'val', transform = val_transform)
    image_ids = train_data.new_image_ids 
    image_path_map = train_data.image_path_map
    #80 objects
    id2object = train_data.id2object
    id2labels = train_data.id2labels
    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                            shuffle = False, num_workers = 4,
                                            pin_memory = True)
    model = MultilabelObject(None, 81).cuda()

    log_dir = "./log/"
    checkpoint = torch.load(os.path.join(log_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    t = tqdm(train_loader, desc = 'Activation')
    count = 0
    yhats = []
    labels = []
    imagefiles = []
    for batch_idx, (images, objects, image_ids) in enumerate(t):

        images = Variable(images).cuda()
        objects = Variable(objects).cuda()        
        object_preds = model(images) 
        m = nn.Sigmoid()
        object_preds_r = m(object_preds)
        count = count + len(image_ids)
        for i in xrange(len(image_ids)):
            image_file_name = image_path_map[image_ids[i]]
            yhat = []
            label = id2labels[image_ids[i]]
            
            for j in xrange(len(object_preds[i])):
                a = object_preds_r[i][j].cpu().data.numpy()
                if a[0] > 0.5:
                    yhat.append(id2object[j])
            yhats.append(yhat)
            labels.append(label)
            imagefiles.append(image_file_name)
        if count % 1000 == 0:
            print("count: " + str(count))
        
    with open('globalyhats_val.pickle', 'wb') as handle:
        pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('globallabels_val.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('imagefiles_val.pickle', 'wb') as handle:
        pickle.dump(imagefiles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    with open('globalyhats.pickle', 'rb') as handle:
        yhats = pickle.load(handle)
    with open('globallabels.pickle', 'rb') as handle:
        labels = pickle.load(handle)
    with open('imagefiles.pickle', 'rb') as handle:
        imagefiles = pickle.load(handle)
    '''

# Get yhats
def get_yhats_test(confidence=0.5):

    
    ann_dir = '/local/yuchi/dataset/coco/annotations'
    image_dir = '/local/yuchi/dataset/coco/'
    crop_size = 224
    image_size = 256
    batch_size = 16
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])

    val_transform = transforms.Compose([ 
        transforms.Scale(image_size),
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(), 
        normalize])

    # Data samplers.
    train_data = CocoObject(ann_dir = ann_dir, image_dir = image_dir, 
        split = 'test', transform = val_transform)
    image_ids = train_data.new_image_ids 
    image_path_map = train_data.image_path_map
    #80 objects
    id2object = train_data.id2object
    id2labels = train_data.id2labels
    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                            shuffle = False, num_workers = 4,
                                            pin_memory = True)
    model = MultilabelObject(None, 81).cuda()

    log_dir = "./log/"
    checkpoint = torch.load(os.path.join(log_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    t = tqdm(train_loader, desc = 'Activation')
    count = 0
    yhats = []
    labels = []
    imagefiles = []
    res = list()
    for batch_idx, (images, objects, image_ids) in enumerate(t):

        images = Variable(images).cuda()
        objects = Variable(objects).cuda()        
        object_preds = model(images) 
        m = nn.Sigmoid()
        object_preds_r = m(object_preds)
        count = count + len(image_ids)
        for i in xrange(len(image_ids)):
            image_file_name = image_path_map[int(image_ids[i])]
            yhat = []
            label = id2labels[int(image_ids[i])]
            
            for j in xrange(len(object_preds[i])):
                a = object_preds_r[i][j].cpu().data.numpy()
                if a > confidence:
                    yhat.append(id2object[j])
            yhats.append(yhat)
            labels.append(label)
            imagefiles.append(image_file_name)
        res.append((image_ids, object_preds.data.cpu(), objects.data.cpu()))
        if count % 1000 == 0:
            print("count: " + str(count))
    
    preds_object   = torch.cat([entry[1] for entry in res], 0)
    targets_object = torch.cat([entry[2] for entry in res], 0)
    eval_score_object = average_precision_score(targets_object.numpy(), preds_object.numpy())
    print('\nmean average precision of object classifier on val data is {}\n'.format(eval_score_object))
    
    with open('globalyhats_test.pickle', 'wb') as handle:
        pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('globallabels_test.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('imagefiles_test.pickle', 'wb') as handle:
        pickle.dump(imagefiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Get yhats
def get_yhats_train(confidence=0.5):

    
    ann_dir = '/home/yuchi/dataset/coco/annotations'
    image_dir = '/home/yuchi/dataset/coco/'
    crop_size = 224
    image_size = 256
    batch_size = 16
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])

    val_transform = transforms.Compose([ 
        transforms.Scale(image_size),
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(), 
        normalize])

    # Data samplers.
    train_data = CocoObject(ann_dir = ann_dir, image_dir = image_dir, 
        split = 'train', transform = val_transform)
    image_ids = train_data.new_image_ids 
    image_path_map = train_data.image_path_map
    #80 objects
    id2object = train_data.id2object
    id2labels = train_data.id2labels
    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                            shuffle = False, num_workers = 4,
                                            pin_memory = True)
    model = MultilabelObject(None, 81).cuda()

    log_dir = "./log/"
    checkpoint = torch.load(os.path.join(log_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    t = tqdm(train_loader, desc = 'Activation')
    count = 0
    yhats = []
    labels = []
    imagefiles = []
    for batch_idx, (images, objects, image_ids) in enumerate(t):

        images = Variable(images).cuda()
        objects = Variable(objects).cuda()        
        object_preds = model(images) 
        m = nn.Sigmoid()
        object_preds_r = m(object_preds)
        count = count + len(image_ids)
        for i in xrange(len(image_ids)):
            image_file_name = image_path_map[image_ids[i]]
            yhat = []
            label = id2labels[image_ids[i]]
            
            for j in xrange(len(object_preds[i])):
                a = object_preds_r[i][j].cpu().data.numpy()
                if a[0] > confidence:
                    yhat.append(id2object[j])
            yhats.append(yhat)
            labels.append(label)
            imagefiles.append(image_file_name)
        if count % 1000 == 0:
            print("count: " + str(count))
        
    with open('globalyhats_train.pickle', 'wb') as handle:
        pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('globallabels_train.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('imagefiles_train.pickle', 'wb') as handle:
        pickle.dump(imagefiles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

   
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_val_transform():
    crop_size = 224
    image_size = 256
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    return transforms.Compose([ 
        transforms.Scale(image_size),
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(), 
        normalize])

def predict_image(image_file):
    with open('id2object.pickle', 'rb') as handle:
        id2object = pickle.load(handle)
    from PIL import Image
    img = Image.open(image_file).convert('RGB')  
    
    val_transform = get_val_transform()
    img = val_transform(img)
    model = MultilabelObject(None, 81).cuda()
    log_dir = "./log/"
    checkpoint = torch.load(os.path.join(log_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    images = img.unsqueeze(0)
    images = Variable(images).cuda()
    #objects = Variable(objects).cuda()
    object_preds = model(images)
    m = nn.Sigmoid()
    object_preds_r = m(object_preds)
    yhat = {}
    for i in xrange(len(object_preds[0])):
        a = object_preds_r[0][i].cpu().data.numpy()
        yhat[id2object[i]] = a[0]
        #print(str(a[0]) + ',' + id2object[i])
    objects = 0
    for key, value in sorted(yhat.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        #print(key + ',' + str(value))
        objects = objects + 1
        if value > 0.5:
            print(key + ',' + str(value))


def get_object_id():
    with open('id2object.pickle', 'rb') as handle:
        id2object = pickle.load(handle)
    with open('object_id_name.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["object_id","object_name"])
        for i in xrange(81):
            csvcontent = []
            csvcontent.append(i)
            csvcontent.append(id2object[i])
            writer.writerow(csvcontent)


def deepinspect(sample_10):
    #compute neuron-feature score
    #feature->neurons mapping
    total_layers = 53
    with open('globalyhats_test.pickle', 'rb') as handle:
        test_yhats = pickle.load(handle)
    with open('globallabels_test.pickle', 'rb') as handle:
        test_labels = pickle.load(handle)
    with open('imagefiles_test.pickle', 'rb') as handle:
        test_imagefiles = pickle.load(handle)

    layer_coverage = [dict() for x in range(total_layers)]
    activation_sum = [dict() for x in range(total_layers)]
    features_sum = {}
    #compute features_sum    
    featuresmap = {}

    infile = open('globalcoveragecocoexp_test_0.5.pickle', 'r')
    processed = 0
    while 1:
        try:
            globalcoverage = pickle.load(infile)
            assert len(globalcoverage) <= 16
            for g in globalcoverage:
                assert len(g["layercoverage"]) == 53
                if processed in sample_10:
                    processed = processed + 1
                    continue
               
                from sets import Set
                
                 
                #for v in list(set(g["jlabel"])):
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

    with open('p_layer_coverage_predicted_90_0.25.pickle', 'wb') as handle:
        pickle.dump(p_layer_coverage, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open('p_layer_coverage_predicted_90_0.25.pickle', 'rb') as handle:
        p_layer_coverage = pickle.load(handle)
    with open('id2object.pickle', 'rb') as handle:
        id2object = pickle.load(handle)



    labels_list = []
    for i in xrange(81):
        labels_list.append(id2object[i])

    #compute pairwise distance
    #distance1 based on probability score
    #distance2 based on sensitivity score
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

    with open('neuron_distance_from_predicted_labels_test_90_0.5.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["object","object", "neuron distance"])
        for key, value in sorted(distance.iteritems(), key=lambda (k,v): (v,k), reverse=False):

            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

def get_10_csv_files(sample_10):
    total_layers = 53
    with open('id2object.pickle', 'rb') as handle:
        id2object = pickle.load(handle)
    with open('globalyhats_test.pickle', 'rb') as handle:
        yhats_raw = pickle.load(handle)
    with open('globallabels_test.pickle', 'rb') as handle:
        labels_raw = pickle.load(handle)
    with open('imagefiles_test.pickle', 'rb') as handle:
        imagefiles_raw = pickle.load(handle)
    yhats = []
    labels = []
    imagefiles = []
    for i in xrange(len(yhats_raw)):
        if i in sample_10:
            yhats.append(yhats_raw[i])
            labels.append(labels_raw[i])
            imagefiles.append(imagefiles_raw[i])
    labels_list = []
    for i in xrange(81):
        labels_list.append(id2object[i])


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
    distance = type2confusion

    with open('objects_directional_type2_confusion_test_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type2 confusion"])
        
 
        for key, value in sorted(type2confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

    with open('objects_directional_type1_confusion_test_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type1 confusion"])
        
 
        for key, value in sorted(type1confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)


    with open('test_labels_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_file","labels"])
        
 
        for i in xrange(len(imagefiles)):
            csvrecord = []
            csvrecord.append(imagefiles[i])
            csvrecord.append(";".join(labels[i]))
            writer.writerow(csvrecord)

    with open('test_predicted_labels_10.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_file","predicted labels"])
        
 
        for i in xrange(len(imagefiles)):
            csvrecord = []
            csvrecord.append(imagefiles[i])
            csvrecord.append(";".join(yhats[i]))
            writer.writerow(csvrecord)

def get_90_csv_files(sample_10):
    total_layers = 53
    with open('id2object.pickle', 'rb') as handle:
        id2object = pickle.load(handle)
    with open('globalyhats_test.pickle', 'rb') as handle:
        yhats_raw = pickle.load(handle)
    with open('globallabels_test.pickle', 'rb') as handle:
        labels_raw = pickle.load(handle)
    with open('imagefiles_test.pickle', 'rb') as handle:
        imagefiles_raw = pickle.load(handle)
    yhats = []
    labels = []
    imagefiles = []
    for i in xrange(len(yhats_raw)):
        if i not in sample_10:
            yhats.append(yhats_raw[i])
            labels.append(labels_raw[i])
            imagefiles.append(imagefiles_raw[i])
    labels_list = []
    for i in xrange(81):
        labels_list.append(id2object[i])


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
    distance = type2confusion

    with open('objects_directional_type2_confusion_test_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["object1","object2", "test data type2 confusion"])
        
 
        for key, value in sorted(type2confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

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
            csvrecord.append(";".join(labels[i]))
            writer.writerow(csvrecord)

    with open('test_predicted_labels_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["image_file","predicted labels"])
        
 
        for i in xrange(len(imagefiles)):
            csvrecord = []
            csvrecord.append(imagefiles[i])
            csvrecord.append(";".join(yhats[i]))
            writer.writerow(csvrecord)

if __name__ == '__main__':
    get_id2object_pkl()
    get_yhats_test()
    get_coverage_test()
    np.random.seed(0)
    sample_10 = np.random.choice(16931, 1693, replace=False)
    deepinspect(sample_10=sample_10)
    get_10_csv_files(sample_10=sample_10)
    get_90_csv_files(sample_10=sample_10)