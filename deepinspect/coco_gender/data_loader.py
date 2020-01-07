import json, os, string, random, time, pickle, gc
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from pycocotools.coco import COCO

class CocoObject(data.Dataset):
    def __init__(self, ann_dir, image_dir, split = 'train', transform = None):
        self.ann_dir = ann_dir
        self.image_dir = image_dir
        self.split = split
        self.transform = transform

        if self.split == 'train':
            ann_path = os.path.join(self.ann_dir, "instances_train2014.json")
            gender_file = "train_0.data"
        else:
            gender_file = "val_0.data"
            ann_path = os.path.join(self.ann_dir, "instances_val2014.json")
        self.cocoAPI = COCO(ann_path)
        self.data = json.load(open(ann_path))
        self.image_ids = [elem['id'] for elem in self.data['images']]
        print(type(self.image_ids[0]))
        if self.split == 'val':
            self.image_ids = self.image_ids[:10000]
        elif self.split == 'test':
            self.image_ids = self.image_ids[10000:]

        self.image_path_map = {elem['id']: elem['file_name'] for elem in self.data['images']}


        genders = pickle.load(open(gender_file))
        imageid2gender = {}
        for gender_image in genders:
            if gender_image['gender'][0] == 1:
                gender_label = 'man'
            else:
                gender_label = 'woman'
            imageid2gender[gender_image["image_id"]] = gender_label
        #80 objects
        id2object = dict()
        object2id = dict()
        self.person_id = -1
        for idx, elem in enumerate(self.data['categories']):
            if elem['name'] == 'person':
                self.person_id = idx
                print("person index: " + str(idx))
                id2object[idx] = "man"
                object2id["man"] = idx
                continue
            id2object[idx] = elem['name']
            object2id[elem['name']] = idx
        id2object[80] = "woman"
        object2id['woman'] = 80
        assert self.person_id != -1
        #generate one-hot encoding objects annotation for every image
        #self.object_ann = np.zeros((len(self.image_ids), 81))
        self.id2object = id2object
        self.object2id = object2id
        self.id2labels = {}
        self.new_image_ids = []
        for idx, image_id in enumerate(self.image_ids):
            ann_ids = self.cocoAPI.getAnnIds(imgIds = image_id)
            anns = self.cocoAPI.loadAnns(ids = ann_ids)
            category_ids = [elem['category_id'] for elem in anns]
            category_names = [elem['name'] for elem in self.cocoAPI.loadCats(ids=category_ids)]
            if 'person' in category_names and image_id not in imageid2gender:
                continue
            else:
                self.new_image_ids.append(image_id)
            #encoding_ids = [object2id[name] for name in category_names]
            #for encoding_id in encoding_ids:
                #self.object_ann[idx, encoding_id] = 1
        print("image ids len: " + str(len(self.image_ids)))
        print("new image ids len: " + str(len(self.new_image_ids)))
        gender_count = {}
        gender_count["man"] = 0
        gender_count["woman"] = 0
        self.object_ann = np.zeros((len(self.new_image_ids), 81))
        for idx, image_id in enumerate(self.new_image_ids):
            ann_ids = self.cocoAPI.getAnnIds(imgIds = image_id)
            anns = self.cocoAPI.loadAnns(ids = ann_ids)
            category_ids = [elem['category_id'] for elem in anns]
            category_names = [elem['name'] for elem in self.cocoAPI.loadCats(ids=category_ids)]
            encoding_ids = []
            self.id2labels[image_id] = []
            for name in category_names:
                if 'person' == name:
                    gender_count[imageid2gender[image_id]] = gender_count[imageid2gender[image_id]] + 1
                    encoding_ids.append(object2id[imageid2gender[image_id]])
                    self.id2labels[image_id].append(imageid2gender[image_id])
                else:
                    encoding_ids.append(object2id[name])
                    self.id2labels[image_id].append(name)
            for encoding_id in encoding_ids:
                self.object_ann[idx, encoding_id] = 1
        print(gender_count)

    def __getitem__(self, index):
        image_id = self.new_image_ids[index]
        image_file_name = self.image_path_map[image_id]
        if self.split == 'train':
            image_path = os.path.join(self.image_dir,"train2014", image_file_name)
        else:
            image_path = os.path.join(self.image_dir,"val2014", image_file_name)

        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.Tensor(self.object_ann[index]), image_id

    def getObjectWeights(self):
        return (self.object_ann == 0).sum(axis = 0) / (1e-8 + self.object_ann.sum(axis = 0))

    def __len__(self):
        return len(self.new_image_ids)