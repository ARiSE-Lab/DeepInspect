import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import euclidean_distances

from util import Test, get_val_and_test_path


def run_trials(datasets, mode):
    print('-'*10, mode, '-'*10)
    for dataset in datasets:
        print('-'*10, dataset, '-'*10)
        mode_str = ''
        if mode == 'MODE':
            if dataset == 'imsitu':
                continue
            mode_str = '_MODE'

        test_folderpath = '../data/'+dataset+'/saved_data_object'
        if not os.path.exists(test_folderpath):
            os.mkdir(test_folderpath)

        test_filepath = test_folderpath+'/'+dataset+'_test'+mode_str
        paths_val, paths_test = get_val_and_test_path(dataset, th_str=th_str)
        if mode == 'MODE':
            paths_test[0] = '../data/'+dataset+'/MODE_dist.csv'

        t = Test(paths_test)
        predicted_label_conf_bias_list, _, _ = t.get_pair_list(t1=0.00)
        np.save(test_filepath, predicted_label_conf_bias_list)



if __name__ == '__main__':
    datasets = ["coco","coco_gender","cifar100","robust_cifar10_small","robust_cifar10_large","robust_cifar10_resnet","imsitu","imagenet"]

    th = 0.5
    th_str = ''
    if th != 0.5:
        th_str = '_'+str(th)

    for saving_dataset in datasets:
        if saving_dataset == 'imsitu':
            continue

        # Get distance matrix
        encoding = 'latin1'
        if saving_dataset in ['cifar100', 'imagenet']:
            encoding = 'ASCII'

        weight_file_name = '../data/'+saving_dataset+'/weight_matrix.pickle'
        with open(weight_file_name, 'rb') as f:
            weight_matrix = pickle.load(f, encoding=encoding)
        d = euclidean_distances(weight_matrix)

        print(saving_dataset)
        print(np.array(weight_matrix).shape)


        # Get mapping between object and id
        if saving_dataset in ['cifar100', 'robust_cifar10_small', 'robust_cifar10_large', 'robust_cifar10_resnet', 'imagenet', 'imsitu']:
            object2id = dict()
            count = 0
            with open('../data/'+saving_dataset+'/neuron_distance_from_predicted_labels_test_90'+th_str+'.csv', 'r') as f_in:
                for line in f_in:
                    tokens = line.split(',')
                    obj1 = tokens[0]
                    obj2 = tokens[1]
                    if obj1 not in ['object', 'object1'] and obj1 not in object2id:
                        object2id[obj1] = count
                        count += 1
                    if obj2 not in ['object', 'object2'] and obj2 not in object2id:
                        object2id[obj2] = count
                        count += 1
            id2object = {object2id[obj]:obj for obj in object2id}
        elif saving_dataset in ['coco', 'coco_gender']:
            with open('../data/'+saving_dataset+'/id2object.pickle', 'rb') as f:
                id2object = pickle.load(f)
            with open('../data/'+saving_dataset+'/object2id.pickle', 'rb') as f:
                object2id = pickle.load(f)

        # Save MODE distance
        with open('../data/'+saving_dataset+'/MODE_dist.csv', 'w') as f_w:
            f_w.write('object,object,neuron distance\n')
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    object1 = str(i)
                    object2 = str(j)
                    object1 = id2object[i]
                    object2 = id2object[j]
                    f_w.write(object1+','+object2+','+str(d[i][j])+'\n')


    run_trials(datasets, 'original')
    run_trials(datasets, 'MODE')
