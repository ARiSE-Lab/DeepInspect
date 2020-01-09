import os, sys
import time
from torch import optim
import random as rand
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tvt
import math
from imsitu import imSituVerbRoleLocalNounEncoder 
from imsitu import imSituTensorEvaluation 
from imsitu import imSituSituation 
from imsitu import imSituSimpleImageFolder
from utils import initLinear
import json
from torchncoverage import NCoverage
import matplotlib.pyplot as plt
import numpy as np
import csv,pickle
import copy


globalcoverage = [] # [{file, label, layercoverage, yhat}]

class vgg_modified(nn.Module):
  def __init__(self):
    super(vgg_modified,self).__init__()
    self.vgg = tv.models.vgg16(pretrained=True)
    self.vgg_features = self.vgg.features
    #self.classifier = nn.Sequential(
            #nn.Dropout(),
    self.lin1 = nn.Linear(512 * 7 * 7, 1024)
    self.relu1 = nn.ReLU(True)
    self.dropout1 = nn.Dropout()
    self.lin2 =  nn.Linear(1024, 1024)
    self.relu2 = nn.ReLU(True)
    self.dropout2 = nn.Dropout()

    initLinear(self.lin1)
    initLinear(self.lin2)
  
  def rep_size(self): return 1024

  def forward(self,x):
    return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))

class resnet_modified_large(nn.Module):
  def __init__(self):
    super(resnet_modified_large, self).__init__()
    self.resnet = tv.models.resnet101(pretrained=True)
    #probably want linear, relu, dropout
    self.linear = nn.Linear(7*7*2048, 1024)
    self.dropout2d = nn.Dropout2d(.5)
    self.dropout = nn.Dropout(.5)
    self.relu = nn.LeakyReLU()
    initLinear(self.linear)

  def base_size(self): return 2048
  def rep_size(self): return 1024

  def forward(self, x):
    x = self.resnet.conv1(x)
    x = self.resnet.bn1(x)
    x = self.resnet.relu(x)
    x = self.resnet.maxpool(x)

    x = self.resnet.layer1(x)
    x = self.resnet.layer2(x)
    x = self.resnet.layer3(x)
    x = self.resnet.layer4(x)
 
    x = self.dropout2d(x)

    #print x.size()
    return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))

  

class resnet_modified_medium(nn.Module):
 def __init__(self):
    super(resnet_modified_medium, self).__init__()
    self.resnet = tv.models.resnet50(pretrained=True)
    #probably want linear, relu, dropout
    self.linear = nn.Linear(7*7*2048, 1024)
    self.dropout2d = nn.Dropout2d(.5)
    self.dropout = nn.Dropout(.5)
    self.relu = nn.LeakyReLU()
    initLinear(self.linear)

 def base_size(self): return 2048
 def rep_size(self): return 1024

 def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
     
        x = self.dropout2d(x)

        #print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
 
 
class resnet_modified_small(nn.Module):
 def __init__(self):
    super(resnet_modified_small, self).__init__()
    self.resnet = tv.models.resnet34(pretrained=True)
    #probably want linear, relu, dropout
    self.linear = nn.Linear(7*7*512, 1024)
    self.dropout2d = nn.Dropout2d(.5)
    self.dropout = nn.Dropout(.5)
    self.relu = nn.LeakyReLU()
    initLinear(self.linear)

 def base_size(self): return 512
 def rep_size(self): return 1024

 def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
     
        x = self.dropout2d(x)

        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
      
class baseline_crf(nn.Module):
   def train_preprocess(self): return self.train_transform
   def dev_preprocess(self): return self.dev_transform

   #these seem like decent splits of imsitu, freq = 0,50,100,282 , prediction type can be "max_max" or "max_marginal"
   def __init__(self, encoding, splits = [50,100,283], prediction_type = "max_max", ngpus = 1, cnn_type = "resnet_101", get_coverage = False):
     super(baseline_crf, self).__init__() 
     
     self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     self.train_transform = tv.transforms.Compose([
            tv.transforms.Scale(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

     self.dev_transform = tv.transforms.Compose([
            tv.transforms.Scale(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

     self.broadcast = []
     self.nsplits = len(splits)
     self.splits = splits
     self.encoding = encoding
     #print(encoding)
     self.prediction_type = prediction_type
     self.n_verbs = encoding.n_verbs()
     self.split_vr = {}
     self.v_roles = {}
     #cnn
     #print cnn_type
     if cnn_type == "resnet_101" : self.cnn = resnet_modified_large()
     elif cnn_type == "resnet_50": self.cnn = resnet_modified_medium()
     elif cnn_type == "resnet_34": self.cnn = resnet_modified_small()
     else: 
       print "unknown base network" 
       exit()

     #self.cnn.resnet.layer1[0].conv2.register_forward_hook(get_neuron_output)
     
     if get_coverage:
       hook_all_conv_layer(self.cnn, get_channel_coverage_group_exp)
     


     #hook_all_layer(self.cnn, get_neuron_output)
     #print(self.cnn.resnet)
     #print(self.cnn)
     self.rep_size = self.cnn.rep_size()
     for s in range(0,len(splits)): self.split_vr[s] = []

     #sort by length
     remapping = []
     for (vr, ns) in encoding.vr_id_n.items(): remapping.append((vr, len(ns)))
     
     #find the right split
     for (vr, l) in remapping:
       i = 0
       for s in splits:
         if l <= s: break
         i+=1  
       _id = (i, vr)
       self.split_vr[i].append(_id)
     total = 0 
     for (k,v) in self.split_vr.items():
       #print "{} {} {}".format(k, len(v), splits[k]*len(v))
       total += splits[k]*len(v) 
     #print "total compute : {}".format(total) 
     
     #keep the splits sorted by vr id, to keep the model const w.r.t the encoding 
     for i in range(0,len(splits)):
       s = sorted(self.split_vr[i], key = lambda x: x[1])
       self.split_vr[i] = []
       #enumerate?
       for (x, vr) in s: 
         _id = (x,len(self.split_vr[i]), vr)
         self.split_vr[i].append(_id)    
         (v,r) = encoding.id_vr[vr]
         if v not in self.v_roles: self.v_roles[v] = []
         self.v_roles[v].append(_id)
    
     #create the mapping for grouping the roles back to the verbs later       
     max_roles = encoding.max_roles()

     #need a list that is nverbs by 6
     self.v_vr = [ 0 for i in range(0, self.encoding.n_verbs()*max_roles) ]
     splits_offset = []
     for i in range(0,len(splits)):
       if i == 0: splits_offset.append(0)
       else: splits_offset.append(splits_offset[-1] + len(self.split_vr[i-1]))
    
     #and we need to compute the position of the corresponding roles, and pad with the 0 symbol
     for i in range(0, self.encoding.n_verbs()):
       offset = max_roles*i
       roles = sorted(self.v_roles[i] , key=lambda x: x[2]) #stored in role order
       self.v_roles[i] = roles
       k = 0
       for (s, pos, r) in roles:
         #add one to account of the 0th element being the padding
         self.v_vr[offset + k] = splits_offset[s] + pos + 1
         k+=1
       #pad
       while k < max_roles:
         self.v_vr[offset + k] = 0
         k+=1
     
     gv_vr = Variable(torch.LongTensor(self.v_vr).cuda())#.view(self.encoding.n_verbs(), -1) 
     for g in range(0,ngpus):
       self.broadcast.append(Variable(torch.LongTensor(self.v_vr).cuda(g)))
     self.v_vr = gv_vr
     #print self.v_vr

     #verb potential
     self.linear_v = nn.Linear(self.rep_size, self.encoding.n_verbs())
     #verb-role-noun potentials
     self.linear_vrn = nn.ModuleList([ nn.Linear(self.rep_size, splits[i]*len(self.split_vr[i])) for i in range(0,len(splits))])
     self.total_vrn = 0
     for i in range(0, len(splits)): self.total_vrn += splits[i]*len(self.split_vr[i])
     print "total encoding vrn : {0}, with padding in {1} groups : {2}".format(encoding.n_verbrolenoun(), self.total_vrn, len(splits))    

     #initilize everything
     initLinear(self.linear_v)
     for _l in self.linear_vrn: initLinear(_l)
     self.mask_args()

   def mask_args(self):
     #go through the and set the weights to negative infinity for out of domain items     
     neg_inf = float("-infinity")
     for v in range(0, self.encoding.n_verbs()):
       for (s, pos, r) in self.v_roles[v]:
         linear = self.linear_vrn[s] 
         #get the offset
#         print self.splits
         start = self.splits[s]*pos+len(self.encoding.vr_n_id[r])
         end = self.splits[s]*(pos+1)
         for k in range(start,end):
           linear.bias.data[k] = -100 #neg_inf
            
   #expects a list of vectors, BxD
   #returns the max index of every vector, max value of each vector and the log_sum_exp of the vector
   def log_sum_exp(self,vec):
     max_score, max_i = torch.max(vec,1)
     max_score_broadcast = max_score.view(-1,1).expand(vec.size())
     return (max_i , max_score,  max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast),1)))

   def forward_max(self, images):
     (_,_,_,_,scores, values) = self.forward(images)
     return (scores, values)

   def forward_features(self, images):
     return self.cnn(images)
   
   def forward(self, image):
     batch_size = image.size()[0]

     rep = self.cnn(image)
     #print self.rep_size
     #print batch_size
     v_potential = self.linear_v(rep)
     
     vrn_potential = []
     vrn_marginal = []
     vr_max = []
     vr_maxi = []
     #first compute the norm
     #step 1 compute the verb-role marginals
     #this loop allows a memory/parrelism tradeoff. 
     #To use less memory but achieve less parrelism, increase the number of groups
     for i,vrn_group in enumerate(self.linear_vrn): 
       #linear for the group
       _vrn = vrn_group(rep).view(-1, self.splits[i])
       
       _vr_maxi, _vr_max ,_vrn_marginal = self.log_sum_exp(_vrn)
       _vr_maxi = _vr_maxi.view(-1, len(self.split_vr[i]))
       _vr_max = _vr_max.view(-1, len(self.split_vr[i]))
       _vrn_marginal = _vrn_marginal.view(-1, len(self.split_vr[i]))
     
       vr_maxi.append(_vr_maxi)
       vr_max.append(_vr_max)
       vrn_potential.append(_vrn.view(batch_size, -1, self.splits[i]))
       vrn_marginal.append(_vrn_marginal)
     
     #concat role groups with the padding symbol 
     zeros = Variable(torch.zeros(batch_size, 1).cuda()) #this is the padding 
     zerosi = Variable(torch.LongTensor(batch_size,1).zero_().cuda())
     vrn_marginal.insert(0, zeros)
     vr_max.insert(0,zeros)
     vr_maxi.insert(0,zerosi)

     #print vrn_marginal
     vrn_marginal = torch.cat(vrn_marginal, 1)
     vr_max = torch.cat(vr_max,1)
     vr_maxi = torch.cat(vr_maxi,1)     

     #print vrn_marginal
     #step 2 compute verb marginals
     #we need to reorganize the role potentials so it is BxVxR
     #gather the marginals in the right way
     v_vr = self.broadcast[torch.cuda.current_device()] 
     vrn_marginal_grouped = vrn_marginal.index_select(1,v_vr).view(batch_size,self.n_verbs,self.encoding.max_roles())
     vr_max_grouped = vr_max.index_select(1,v_vr).view(batch_size, self.n_verbs, self.encoding.max_roles()) 
     vr_maxi_grouped = vr_maxi.index_select(1,v_vr).view(batch_size, self.n_verbs, self.encoding.max_roles())
     
     # product ( sum since we are in log space )
     v_marginal = torch.sum(vrn_marginal_grouped, 2).view(batch_size, self.n_verbs) + v_potential
    
     #step 3 compute the final sum over verbs
     _, _ , norm  = self.log_sum_exp(v_marginal)
     #compute the maxes

     #max_max probs
     v_max = torch.sum(vr_max_grouped,2).view(batch_size, self.n_verbs) + v_potential #these are the scores
     #we don't actually care, we want a max prediction per verb
     #max_max_vi , max_max_v_score = max(v_max,1)
     #max_max_prob = exp(max_max_v_score - norm)
     #max_max_vrn_i = vr_maxi_grouped.gather(1,max_max_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))

     #offset so we can use index select... is there a better way to do this?
     #max_marginal probs 
     #max_marg_vi , max_marginal_verb_score = max(v_marginal, 1)
     #max_marginal_prob = exp(max_marginal_verb_score - norm)
     #max_marg_vrn_i = vr_maxi_grouped.gather(1,max_marg_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))
     
     #this potentially does not work with parrelism, in which case we should figure something out 
     if self.prediction_type == "max_max":
       rv = (rep, v_potential, vrn_potential, norm, v_max, vr_maxi_grouped) 
     elif self.prediction_type == "max_marginal":
       rv = (rep, v_potential, vrn_potential, norm, v_marginal, vr_maxi_grouped) 
     else:
       print "unkown inference type"
       rv = ()
     return rv

  
   #computes log( (1 - exp(x)) * (1 - exp(y)) ) =  1 - exp(y) - exp(x) + exp(y)*exp(x) = 1 - exp(V), so V=  log(exp(y) + exp(x) - exp(x)*exp(y))
   #returns the the log of V 
   def logsumexp_nx_ny_xy(self, x, y):
     #_,_, v = self.log_sum_exp(torch.cat([x, y, torch.log(torch.exp(x+y))]).view(1,3))
     if x > y: 
       return torch.log(torch.exp(y-x) + 1 - torch.exp(y) + 1e-8) + x
     else:
       return torch.log(torch.exp(x-y) + 1 - torch.exp(x) + 1e-8) + y

   def sum_loss(self, v_potential, vrn_potential, norm, situations, n_refs):
     #compute the mil losses... perhaps this should be a different method to facilitate parrelism?
     batch_size = v_potential.size()[0]
     mr = self.encoding.max_roles()
     for i in range(0,batch_size):
       _norm = norm[i]
       _v = v_potential[i]
       _vrn = []
       _ref = situations[i]
       for pot in vrn_potential: _vrn.append(pot[i])
       for r in range(0,n_refs):
         v = _ref[0]
         pots = _v[v]
         for (pos,(s, idx, rid)) in enumerate(self.v_roles[v]):
           pots = pots + _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
         if pots.data[0] > _norm.data[0]: 
           print "inference error"
           print pots
           print _norm
         if i == 0 and r == 0: loss = pots-_norm
         else: loss = loss + pots - _norm
     return -loss/(batch_size*n_refs)

   def mil_loss(self, v_potential, vrn_potential, norm,  situations, n_refs): 
     #compute the mil losses... perhaps this should be a different method to facilitate parrelism?
     batch_size = v_potential.size()[0]
     mr = self.encoding.max_roles()
     for i in range(0,batch_size):
       _norm = norm[i]
       _v = v_potential[i]
       _vrn = []
       _ref = situations[i]
       for pot in vrn_potential: _vrn.append(pot[i])
       for r in range(0,n_refs):
         v = _ref[0]
         pots = _v[v]
         for (pos,(s, idx, rid)) in enumerate(self.v_roles[v]):
       #    print _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
#_vrn[s][idx][
           pots = pots + _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
         if pots.data[0] > _norm.data[0]: 
           print "inference error"
           print pots
           print _norm
         if r == 0: _tot = pots-_norm 
         else : _tot = self.logsumexp_nx_ny_xy(_tot, pots-_norm)
       if i == 0: loss = _tot
       else: loss = loss + _tot
     return -loss/batch_size



def format_dict(d, s, p):
    rv = ""
    for (k,v) in d.items():
      if len(rv) > 0: rv += " , "
      rv+=p+str(k) + ":" + s.format(v*100)
    return rv

def predict_human_readable (dataset_loader, simple_dataset,  model, outdir, top_k):
  model.eval()  
  print "predicting..." 
  mx = len(dataset_loader) 
  for i, (input, index) in enumerate(dataset_loader):
      print "{}/{} batches".format(i+1,mx)
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      #print(predictions)
      #(s_sorted, idx) = torch.sort(scores, 1, True)
      human = encoder.to_situation(predictions)
      print(human)
      (b,p,d) = predictions.size()
      for _b in range(0,b):
        items = []
        offset = _b *p
        for _p in range(0, p):
          items.append(human[offset + _p])
          items[-1]["score"] = scores.data[_b][_p]
        items = sorted(items, key = lambda x: -x["score"])[:top_k]
        name = simple_dataset.images[index[_b][0]].split(".")[:-1]
        name.append("predictions")
        outfile = outdir + ".".join(name)
        json.dump(items,open(outfile,"w"))


def compute_features(dataset_loader, simple_dataset,  model, outdir):
  model.eval()  
  print "computing features..." 
  mx = len(dataset_loader) 
  for i, (input, index) in enumerate(dataset_loader):
      print "{}/{} batches\r".format(i+1,mx) ,
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      features  = model.forward_features(input_var).cpu().data
      b = index.size()[0]
      for _b in range(0,b):
        name = simple_dataset.images[index[_b][0]].split(".")[:-1]
        name.append("features")
        outfile = outdir + ".".join(name)
        torch.save(features[_b], outfile)
  print "\ndone."

def eval_model(dataset_loader, encoding, model):
    model.eval()
    print "evaluating model..."
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
 
    mx = len(dataset_loader) 
    for i, (index, input, target) in enumerate(dataset_loader):
      print "{}/{} batches\r".format(i+1,mx) ,
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      target_var = torch.autograd.Variable(target.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      (s_sorted, idx) = torch.sort(scores, 1, True)
      top1.add_point(target, predictions.data, idx.data)
      top5.add_point(target, predictions.data, idx.data)
      
    print "\ndone."
    return (top1, top5) 

def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, save_dir, timing = False): 
    model.train()

    time_all = time.time()

    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
    loss_total = 0 
    print_freq = 10
    total_steps = 0
    avg_scores = []
  
    for k in range(0,max_epoch):  
      for i, (index, input, target) in enumerate(train_loader):
        total_steps += 1
   
        t0 = time.time()
        t1 = time.time() 
      
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
        (_,v,vrn,norm,scores,predictions)  = pmodel(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        #print norm 
        if timing : print "forward time = {}".format(time.time() - t1)
        optimizer.zero_grad()
        t1 = time.time()
        loss = model.mil_loss(v,vrn,norm, target, 3)
        if timing: print "loss time = {}".format(time.time() - t1)
        t1 = time.time()
        loss.backward()
        #print loss
        if timing: print "backward time = {}".format(time.time() - t1)
        optimizer.step()
        loss_total += loss.data[0]
        #score situation
        t2 = time.time() 
        top1.add_point(target, predictions.data, idx.data)
        top5.add_point(target, predictions.data, idx.data)
     
        if timing: print "eval time = {}".format(time.time() - t2)
        if timing: print "batch time = {}".format(time.time() - t0)
        if total_steps % print_freq == 0:
           top1_a = top1.get_average_results()
           top5_a = top5.get_average_results()
           print "{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(total_steps-1,k,i, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a,"{:.2f}","5-"), loss.data[0], loss_total / ((total_steps-1)%eval_frequency) , (time.time() - time_all)/ ((total_steps-1)%eval_frequency))
        if total_steps % eval_frequency == 0:
          print "eval..."    
          etime = time.time()
          (top1, top5) = eval_model(dev_loader, encoding, model)
          model.train() 
          print "... done after {:.2f} s".format(time.time() - etime)
          top1_a = top1.get_average_results()
          top5_a = top5.get_average_results()

          avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
          avg_score /= 8

          print "Dev {} average :{:.2f} {} {}".format(total_steps-1, avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-"))
          
          avg_scores.append(avg_score)
          maxv = max(avg_scores)

          if maxv == avg_scores[-1]: 
            torch.save(model.state_dict(), save_dir + "/{0}.model".format(maxv))   
            print "new best model saved! {0}".format(maxv)

          top1 = imSituTensorEvaluation(1, 3, encoding)
          top5 = imSituTensorEvaluation(5, 3, encoding)
          loss_total = 0
          time_all = time.time()

def get_neuron_output(self, input, output):
  # input is a tuple of packed inputs
  # output is a Variable. output.data is the Tensor we are interested
  print('Inside ' + self.__class__.__name__ + ' forward')
  print('Inside ' + str(self) + ' forward')
  print('input: ', type(input))
  print('input[0]: ', type(input[0]))
  print('output: ', type(output))

  print('input size:', input[0].size())
  print('output size:', output.data.size())
  print('output norm:', output.data.norm())
  print('output.shape[0] shape:', output.data[0].size())
  print('len(output[0]):', len(output.data[0]))
  print('-----------------------------------------')

def get_channel_coverage(self, input, output):

  nc = NCoverage(threshold = 0.5)
  print('Layer: ' + str(self))
  covered_channel = nc.get_channel_coverage(output.data)
  print('total number of channels: ' + str(len(output.data[0])))
  print('covered channels: ' + str(len(covered_channel)))
  print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))

def get_channel_coverage_exp(self, input, output):
  global globalcoverage
  nc = NCoverage(threshold = 0.5)
  print('Layer: ' + str(self))
  covered_channel = nc.get_channel_coverage(output.data)
  #print('total number of channels: ' + str(len(output.data[0])))
  #print('covered channels: ' + str(len(covered_channel)))
  print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
  globalcoverage[-1]["layercoverage"].append((len(output.data[0]), covered_channel))
  #print(covered_channel)

def get_channel_coverage_group_exp(self, input, output):
  global globalcoverage
  nc = NCoverage(threshold = 0.5)
  #print('Layer: ' + str(self))
  covered_channel_group = nc.get_channel_coverage_group(output.data)
  for c in xrange(len(covered_channel_group)):
    #print(c)
    d = -1*(c+1)
    if "layercoverage" in globalcoverage[d]:
      if len(globalcoverage[d]["layercoverage"]) > 35:
        print("error getting coverage: > 35")
    else:
      globalcoverage[d]["layercoverage"] = []
    #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
    covered_channel = covered_channel_group[d]
    globalcoverage[d]["layercoverage"].append((len(output.data[0]), covered_channel))
  #print('total number of channels: ' + str(len(output.data[0])))
  #print('covered channels: ' + str(len(covered_channel)))
  #print('covered percentage: ' + str(len(covered_channel)*1.0/len(output.data[0])))
  #globalcoverage[-1]["layercoverage"].append((len(output.data[0]), covered_channel))

def hook_all_layer(net, handler):
    for l in net._modules.keys():
        net._modules.get(l).register_forward_hook(handler)
        hook_all_layer(net._modules.get(l), handler)

def hook_all_conv_layer(net, handler):
    for l in net._modules.keys():
        if isinstance(net._modules.get(l), torch.nn.modules.conv.Conv2d):
            net._modules.get(l).register_forward_hook(handler)
        hook_all_conv_layer(net._modules.get(l), handler)

class imSituSimpleImage(data.Dataset):
 # partially borrowed from ImageFolder dataset, but eliminating the assumption about labels
   def is_image_file(self,filename):
    return any(filename.endswith(extension) for extension in self.ext)  
  
   def __init__(self, file, transform=None):
        self.transform = transform
        #list all images        
        self.ext = [ '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        self.images = [file]
 
   def __getitem__(self, index):
        _id = self.images[index]
        img = Image.open(_id).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, torch.LongTensor([index])

   def __len__(self):
        return len(self.images)

class imSituSimpleImageFolder1(data.Dataset):
 # partially borrowed from ImageFolder dataset, but eliminating the assumption about labels
   def is_image_file(self,filename):
    return any(filename.endswith(extension) for extension in self.ext)  
  
   def get_images(self,dir):
    images = []
    for target in os.listdir(dir):
        f = os.path.join(dir, target)
        if os.path.isdir(f):
            continue
        if self.is_image_file(f):
          images.append(target)
    return images

   def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        #list all images        
        self.ext = [ '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        self.images = self.get_images(root)
 
   def __getitem__(self, index):
        _id = os.path.join(self.root,self.images[index])
        img = Image.open(_id).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, torch.LongTensor([index]), _id

   def __len__(self):
        return len(self.images)

class imSituSimpleImageFolder2(data.Dataset):
 # partially borrowed from ImageFolder dataset, but eliminating the assumption about labels
   def is_image_file(self,filename):
    return any(filename.endswith(extension) for extension in self.ext)  
   def __init__(self, root, annotation, transform=None):
        self.root = root
        self.transform = transform
        #list all images        
        self.ext = [ '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        #self.images = self.get_images(root)
        self.imsitu = annotation
        self.images = list(self.imsitu.keys())
 
   def __getitem__(self, index):
        _id = os.path.join(self.root,self.images[index])
        img = Image.open(_id).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, torch.LongTensor([index]), self.images[index]

   def __len__(self):
        return len(self.images)

class imSituSituation1(data.Dataset):
   def __init__(self, root, annotation_file, encoder, transform=None):
        self.root = root
        self.imsitu = annotation_file
        self.ids = list(self.imsitu.keys())
        self.encoder = encoder
        self.transform = transform
   
   def index_image(self, index):
        rv = []
        index = index.view(-1)
        for i in range(index.size()[0]):
          rv.append(self.ids[index[i]])
        return rv
      
   def __getitem__(self, index):
        imsitu = self.imsitu
        _id = self.ids[index]
        ann = self.imsitu[_id]
       
        img = Image.open(os.path.join(self.root, _id)).convert('RGB')
        
        if self.transform is not None: img = self.transform(img)
        target = self.encoder.to_tensor([ann])

        return (torch.LongTensor([index]), img, target)

   def __len__(self):
        return len(self.ids)

def predict_image(file):
  imsitu = json.load(open("imsitu_space.json"))
  nouns = imsitu["nouns"]
  verbs = imsitu["verbs"]
  encoding_file = "baseline_models/baseline_encoder"
  encoder = torch.load(encoding_file)

  model = baseline_crf(encoder, cnn_type = "resnet_34")

  weights_file = "baseline_models/baseline_resnet_34"
  model.load_state_dict(torch.load(weights_file))
  model.cuda()

  folder_dataset = imSituSimpleImage(file, model.dev_preprocess())
  image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = 64, shuffle = False, num_workers = 3) 
  top_k = 1
  #predict_human_readable(image_loader, folder_dataset, model, 64, top_k)
  model.eval()  
  print "predicting..." 
  mx = len(image_loader) 
  for i, (input, index) in enumerate(image_loader):
      print "{}/{} batches".format(i+1,mx)
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      #(s_sorted, idx) = torch.sort(scores, 1, True)
      human = encoder.to_situation(predictions)
      (b,p,d) = predictions.size()
      for _b in range(0,b):
        items = []
        offset = _b *p
        for _p in range(0, p):
          items.append(human[offset + _p])
          items[-1]["score"] = scores.data[_b][_p]
        items = sorted(items, key = lambda x: -x["score"])[:top_k]
        name = folder_dataset.images[index[_b][0]].split(".")[:-1]
        name.append("predictions")
        print(".".join(name) + " : ")
        #outfile = outdir + ".".join(name)
        #json.dump(items,open(outfile,"w"))
        print(items)

        for k in items[0]["frames"][0].keys():
          v = items[0]["frames"][0][k]
          if v in nouns:
            print(v)
            items[0]["frames"][0][k] = nouns[v]["gloss"]
        print(items)

def predict_image_group(folder):
  imsitu = json.load(open("imsitu_space.json"))
  nouns = imsitu["nouns"]
  verbs = imsitu["verbs"]
  encoding_file = "baseline_models/baseline_encoder"
  encoder = torch.load(encoding_file)

  model = baseline_crf(encoder, cnn_type = "resnet_34")

  weights_file = "baseline_models/baseline_resnet_34"
  model.load_state_dict(torch.load(weights_file))
  model.cuda()

  folder_dataset = imSituSimpleImageFolder1(folder, model.dev_preprocess())
  image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = 64, shuffle = False, num_workers = 3) 
  top_k = 1
  #predict_human_readable(image_loader, folder_dataset, model, 64, top_k)
  model.eval()  
  print "predicting..." 
  mx = len(image_loader) 
  for i, (input, index, id) in enumerate(image_loader):
      print("debug")
      print(id)
      print "{}/{} batches".format(i+1,mx)
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      #(s_sorted, idx) = torch.sort(scores, 1, True)
      human = encoder.to_situation(predictions)
      (b,p,d) = predictions.size()
      for _b in range(0,b):
        items = []
        offset = _b *p
        for _p in range(0, p):
          items.append(human[offset + _p])
          items[-1]["score"] = scores.data[_b][_p]
        items = sorted(items, key = lambda x: -x["score"])[:top_k]
        
        name = folder_dataset.images[index[_b][0]].split(".")[:-1]
        
        name.append("predictions")
        print(".".join(name) + " : ")
        #outfile = outdir + ".".join(name)
        #json.dump(items,open(outfile,"w"))
        print(items)

        for k in items[0]["frames"][0].keys():
          v = items[0]["frames"][0][k]
          if v in nouns:
            #print(v)
            items[0]["frames"][0][k] = nouns[v]["gloss"]
        print(items)

def predict_image_1(file):
  imsitu = json.load(open("imsitu_space.json"))
  nouns = imsitu["nouns"]
  verbs = imsitu["verbs"]
  encoding_file = "baseline_models/baseline_encoder"
  encoder = torch.load(encoding_file)

  model = baseline_crf(encoder, cnn_type = "resnet_34")

  weights_file = "baseline_models/baseline_resnet_34"
  model.load_state_dict(torch.load(weights_file))
  model.cuda()

  img = Image.open(file).convert('RGB')
  img = model.dev_preprocess()(img)
  #folder_dataset = imSituSimpleImage(file, model.dev_preprocess())
  #image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = 64, shuffle = False, num_workers = 3) 
  top_k = 1
  #predict_human_readable(image_loader, folder_dataset, model, 64, top_k)
  model.eval()  
  print "predicting..." 
  input = img.unsqueeze(0)
  print(input.size())
  input_var = torch.autograd.Variable(input.cuda(), volatile = True)
  (scores,predictions)  = model.forward_max(input_var)
  #(s_sorted, idx) = torch.sort(scores, 1, True)
  human = encoder.to_situation(predictions)
  (b,p,d) = predictions.size()
  #print("b,p,d")
  #print(b)
  
  items = []
  offset = 0 *p
  for _p in range(0, p):
    items.append(human[offset + _p])
    items[-1]["score"] = scores.data[0][_p]
  items = sorted(items, key = lambda x: -x["score"])[:top_k]
  print(file + " : ")
  #outfile = outdir + ".".join(name)
  #json.dump(items,open(outfile,"w"))
  print(items)
  copyitems = items
  for k in items[0]["frames"][0].keys():
    v = items[0]["frames"][0][k]
    if v in nouns:
      print(v)
      items[0]["frames"][0][k] = nouns[v]["gloss"]
  print(items)


def get_coverage_test(folder, datasetname):
  global globalcoverage
  datasetjson = json.load(open(datasetname))
  imsitu = json.load(open("imsitu_space.json"))
  nouns = imsitu["nouns"]
  verbs = imsitu["verbs"]
  encoding_file = "baseline_models/baseline_encoder"
  encoder = torch.load(encoding_file)

  model = baseline_crf(encoder, cnn_type = "resnet_34", get_coverage = True)

  weights_file = "baseline_models/baseline_resnet_34"
  model.load_state_dict(torch.load(weights_file))
  model.cuda()

  folder_dataset = imSituSimpleImageFolder1(folder, model.dev_preprocess())
  image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = 64, shuffle = False, num_workers = 3) 
  top_k = 1
  #predict_human_readable(image_loader, folder_dataset, model, 64, top_k)
  model.eval()  
  print "predicting..." 
  mx = len(image_loader)
  count = 0
  for i, (input, index, image_files) in enumerate(image_loader):
    print("debug")
    print(image_files)
    count = count + len(image_files)
    
    for image_file in image_files:
      image_name = os.path.basename(image_file)
      globalcoverage.append({})
      globalcoverage[-1]["file"] = image_file
      globalcoverage[-1]["label"] = ""
      globalcoverage[-1]["dataset"] = datasetname
      globalcoverage[-1]["jlabel"] = datasetjson[image_name]
    print "{}/{} batches".format(i+1,mx)
    input_var = torch.autograd.Variable(input.cuda(), volatile = True)
    (scores,predictions)  = model.forward_max(input_var)
    #(s_sorted, idx) = torch.sort(scores, 1, True)
    human = encoder.to_situation(predictions)
    print("count: " + str(count))
    '''
    (b,p,d) = predictions.size()
    for _b in range(0,b):
      items = []
      offset = _b *p
      for _p in range(0, p):
        items.append(human[offset + _p])
        items[-1]["score"] = scores.data[_b][_p]
      items = sorted(items, key = lambda x: -x["score"])[:top_k]
      
      name = folder_dataset.images[index[_b][0]].split(".")[:-1]
      
      name.append("predictions")
      print(".".join(name) + " : ")
      #outfile = outdir + ".".join(name)
      #json.dump(items,open(outfile,"w"))
      print(items)

      for k in items[0]["frames"][0].keys():
        v = items[0]["frames"][0][k]
        if v in nouns:
          #print(v)
          items[0]["frames"][0][k] = nouns[v]["gloss"]
      print(items)
    '''
  import pickle
  with open('globalcoverageexp12.pickle', 'wb') as handle:
    pickle.dump(globalcoverage, handle, protocol=pickle.HIGHEST_PROTOCOL)


def extract_test_data_coverage(dataset_path):
  from shutil import copyfile
  global globalcoverage
  msyn = ["n10745332","n10288763","n02472293","n02472987","n10289039","n10582746","n10289176","n10287213","n03716327","n10288516"]
  wsyn= ["n10787470","n08477634","n09911226","n10788852","n10787470"]
  #msyn = ["n10287213"]
  #wsyn = ["n10787470"]

  debug_count = 0
  di = "test.json"

    #verbs_num = 0
  man_num = 0
  woman_num = 0
  test = json.load(open(di))
  for i in test.keys():
    #print(i)
    debug_count = debug_count + 1

    src_file = dataset_path + i
    dst_file = "resized_256_test/" + i
    copyfile(src_file, dst_file)

  print("debug_count " + str(debug_count))
  get_coverage_test("resized_256_test/","test.json")

def get_test_data_yhats_labels_top100(folder="./resized_256_test/",datasetname="test.json"):
  foi = []
  fi = 0
  with open('featurelist.csv', 'rb') as csvfile:
    features_of_interest = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(features_of_interest)
    for f in features_of_interest:
      foi.append(f[0])
      fi = fi + 1
      if fi > 100:
        break
  datasetjson = json.load(open(datasetname))
  imsitu = json.load(open("imsitu_space.json"))
  nouns = imsitu["nouns"]
  verbs = imsitu["verbs"]
  encoding_file = "baseline_models/baseline_encoder"
  encoder = torch.load(encoding_file)

  model = baseline_crf(encoder, cnn_type = "resnet_34")

  weights_file = "baseline_models/baseline_resnet_34"
  model.load_state_dict(torch.load(weights_file))
  model.cuda()

  folder_dataset = imSituSimpleImageFolder2(folder, datasetjson, model.dev_preprocess())
  image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = 64, shuffle = False, num_workers = 3) 
  top_k = 1
  #predict_human_readable(image_loader, folder_dataset, model, 64, top_k)
  model.eval()  
  print "predicting..." 
  mx = len(image_loader)
  count = 0
  yhats = []
  labels = []
  imagefiles = []
  for i, (input, index, image_files) in enumerate(image_loader):
    #print("debug")
    print(len(image_files))
    imagefiles = imagefiles + list(image_files)
    count = count + len(image_files)
    print "{}/{} batches".format(i+1,mx)
    input_var = torch.autograd.Variable(input.cuda(), volatile = True)
    (scores,predictions)  = model.forward_max(input_var)
    #(s_sorted, idx) = torch.sort(scores, 1, True)
    human = encoder.to_situation(predictions)
    print("count: " + str(count))

    for image_file in image_files:
      label = []
      jlabel = datasetjson[image_file]
      for s in xrange(3):
          for k,v in jlabel[u"frames"][s].iteritems():
            if v in nouns:
              v = nouns[v]["gloss"][0]
              if v in foi and v not in label:
                label.append(v)
      labels.append(label)
    (b,p,d) = predictions.size()
    assert b == len(image_files)
    for _b in range(0,b):
      items = []
      offset = _b *p
      for _p in range(0, p):
        items.append(human[offset + _p])
        items[-1]["score"] = scores.data[_b][_p]
      items = sorted(items, key = lambda x: -x["score"])[:top_k]
      
      name = folder_dataset.images[index[_b][0]].split(".")[:-1]
      
      name.append("predictions")
      #print(".".join(name) + " : ")
      #outfile = outdir + ".".join(name)
      #json.dump(items,open(outfile,"w"))
      #print(items)
      yhat = []
      for k in items[0]["frames"][0].keys():
        v = items[0]["frames"][0][k]
        if v in nouns and nouns[v]["gloss"][0] in foi and nouns[v]["gloss"][0] not in yhat:
          yhat.append(nouns[v]["gloss"][0])
      yhats.append(yhat)
  print(len(imagefiles))
  print(len(yhats))
  print(len(labels))
  with open('globalyhats_test.pickle', 'wb') as handle:
    pickle.dump(yhats, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('globallabels_test.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('imagefiles_test.pickle', 'wb') as handle:
    pickle.dump(imagefiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deepinspect():
  import pickle
  with open('globalyhats_test.pickle', 'rb') as handle:
    test_yhats = pickle.load(handle)
  with open('globallabels_test.pickle', 'rb') as handle:
    test_labels = pickle.load(handle)
  with open('imagefiles_test.pickle', 'rb') as handle:
    test_imagefiles = pickle.load(handle)

  idconvert = {}
  for i in xrange(len(test_imagefiles)):
    idconvert["resized_256_test/" + test_imagefiles[i]] = i
  #msyn = ["n10287213"]
  #wsyn = ["n10787470"]
  msyn = ["n10745332","n10288763","n02472293","n02472987","n10289039","n10582746","n10289176","n10287213","n03716327","n10288516"]
  wsyn= ["n10787470","n08477634","n09911226","n10788852","n10787470"]

  imsitu = json.load(open("imsitu_space.json"))
  nouns = imsitu["nouns"]
  verbs = imsitu["verbs"]

  layer_coverage = [dict() for x in range(36)]

  import csv
  #read top 100 features
  foi = []
  fi = 0
  with open('featurelist.csv', 'rb') as csvfile:
    features_of_interest = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(features_of_interest)
    for f in features_of_interest:
      foi.append(f[0])
      fi = fi + 1
      if fi > 100:
        break
  #print(foi)

  import pickle
  with open('globalcoverageexp12.pickle', 'rb') as handle:
    globalcoverage = pickle.load(handle)
  
  #feature->neurons mapping
  featuresmap = {}
  from sets import Set

  #compute neuron-feature score


  total_layers = len(globalcoverage[0]["layercoverage"])
  print("total layers: " + str(total_layers))
  
  layer_coverage = [dict() for x in range(total_layers)]
  features_sum = {}
  #compute features_sum
  uflag = {} #unique_flag
  count_woman_outdoors = 0
  count_man_outdoors = 0
  total_images = 0
  '''
  '''
  #write objects into featurelist.csv
  for r in globalcoverage:
    if r["dataset"] == "test.json": 
      total_images = total_images + 1
      uflag = {} #unique_flag
      #print(test_imagefiles[idconvert[r['file']]])
      #print(r['file'])
      #assert test_imagefiles[idconvert[r['file']]] == r['file']
      for v in test_yhats[idconvert[r['file']]]:
        if v not in uflag:
          if v not in features_sum.keys():
            features_sum[v] = 1
            uflag[v] = 1
          else:
            features_sum[v] = features_sum[v] + 1
            uflag[v] = 1

      if total_images%2000 == 0:
        print("process: " + str(total_images))

  #compute features scores
  print("compute features' scores")

  for r in globalcoverage:
    if r["dataset"] == "test.json": 
      total_images = total_images + 1
      uflag = {} #unique_flag
      #print(test_imagefiles[total_images-1])
      #print(r['file'])
      #assert test_imagefiles[idconvert[r['file']]] == r['file']
      for v in test_yhats[idconvert[r['file']]]:

        if v not in uflag:
          if v not in features_sum.keys():
            features_sum[v] = 1
            uflag[v] = 1
          else:
            features_sum[v] = features_sum[v] + 1
            uflag[v] = 1

      if total_images%2000 == 0:
        print("process: " + str(total_images))

  process_count = 0
  total_count = len(globalcoverage)
  for r in globalcoverage:
    #print("process count: " + str(process_count))
    #print("total: " + str(total_count))
    if r["dataset"] == "test.json":
      featurelist = {} #unique_feature_list
      #total_training_set = total_training_set + 1
      #assert test_imagefiles[idconvert[r['file']]] == r['file']
      for v in test_yhats[idconvert[r['file']]]:
        featurelist[v] = 1
      for l in xrange(total_layers):
        for j in r["layercoverage"][l][1]:
          if j not in layer_coverage[l].keys():
            layer_coverage[l][j] = {}
          for v in featurelist.keys():
            if v not in layer_coverage[l][j].keys():
              layer_coverage[l][j][v] = 1
            else:
              layer_coverage[l][j][v] = layer_coverage[l][j][v] + 1
    process_count = process_count + 1
    if process_count%1000 == 0:
        print("process: " + str(process_count))
  #compute activation_sum
  activation_sum = [dict() for x in range(total_layers)]
  for r in globalcoverage:
    if r["dataset"] == "test.json": 
      for l in xrange(total_layers):
        for j in r["layercoverage"][l][1]:
          if j not in activation_sum[l].keys():
            activation_sum[l][j] = 1
          else:
            activation_sum[l][j] = activation_sum[l][j] + 1


  #nomalize score(feature probability with respect to each neuron)
  p_layer_coverage = copy.deepcopy(layer_coverage)

  for l in xrange(total_layers):
    for n in layer_coverage[l].keys():
      for v in layer_coverage[l][n].keys():
        p_layer_coverage[l][n][v] = layer_coverage[l][n][v]*1.0 / features_sum[v]



  
  with open('p_layer_coverage_exp12.pickle', 'wb') as handle:
    pickle.dump(p_layer_coverage, handle, protocol=pickle.HIGHEST_PROTOCOL)
  

  total_layers = 36
  with open('p_layer_coverage_exp12.pickle', 'rb') as handle:
    p_layer_coverage = pickle.load(handle)


  print("compute distance")
  labels_list = foi;
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
  with open('neuron_distance_from_predicted_labels_test_90.csv', 'wb',0) as csvfile1:
        writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["object","object", "neuron distance"])
        for key, value in sorted(distance1.iteritems(), key=lambda (k,v): (v,k), reverse=False):

            csvrecord = []
            csvrecord.append(key[0])
            csvrecord.append(key[1])
            csvrecord.append(value)
            writer.writerow(csvrecord)

def generate_csv_test():
  with open('globalyhats_test.pickle', 'rb') as handle:
    yhats = pickle.load(handle)
  with open('globallabels_test.pickle', 'rb') as handle:
    labels = pickle.load(handle)
  with open('imagefiles_test.pickle', 'rb') as handle:
    imagefiles = pickle.load(handle)
  with open('imsitu_test_labels_top100objects.csv', 'wb',0) as csvfile1:
    writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(["image_file","labels"]) 

    for i in xrange(len(imagefiles)):
      csvrecord = []
      csvrecord.append(imagefiles[i])
      csvrecord.append(";".join(set(labels[i])))
      writer.writerow(csvrecord)

  with open('imsitu_test_predicted_labels_top100objects.csv', 'wb',0) as csvfile1:
    writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(["image_file","predicted labels"])
    for i in xrange(len(imagefiles)):
      csvrecord = []
      csvrecord.append(imagefiles[i])
      csvrecord.append(";".join(set(yhats[i])))
      writer.writerow(csvrecord)

  object_count = {}
    
  for r in yhats:
      
      for v in list(set(r)):

          if v not in object_count.keys():
              object_count[v] = 1
          else:
              object_count[v] = object_count[v] + 1


  pairwise_object_count = {}

  for r in yhats:
     
      number_of_object = len(list(set(r)))
      objectslist = list(set(r))
      if number_of_object <= 1:
          continue
      
      for i in range(0, number_of_object):
          for j in range(i+1, number_of_object):

              if (objectslist[i],objectslist[j]) in pairwise_object_count.keys():
                  pairwise_object_count[(objectslist[i],objectslist[j])] = pairwise_object_count[(objectslist[i],objectslist[j])] + 1
                  pairwise_object_count[(objectslist[j],objectslist[i])] = pairwise_object_count[(objectslist[j],objectslist[i])] + 1
              else:
                  pairwise_object_count[(objectslist[i],objectslist[j])] = 1
                  pairwise_object_count[(objectslist[j],objectslist[i])] = 1
  with open('concurrence_count_90.csv', 'wb',0) as csvfile1:
      writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      
      writer.writerow(["object1","object2", "concurrence count"])
      

      for key, value in sorted(pairwise_object_count.iteritems(), key=lambda (k,v): (v,k), reverse=False):
          csvrecord = []
          csvrecord.append(key[0])
          csvrecord.append(key[1])
          csvrecord.append(value)
          writer.writerow(csvrecord)

def output_two_types_confusion():
  foi = []
  fi = 0
  with open('featurelist.csv', 'rb') as csvfile:
    features_of_interest = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(features_of_interest)
    for f in features_of_interest:
      foi.append(f[0])
      fi = fi + 1
      if fi > 100:
        break
  with open('globalyhats_test.pickle', 'rb') as handle:
    yhats = pickle.load(handle)
  with open('globallabels_test.pickle', 'rb') as handle:
    labels = pickle.load(handle)
  with open('imagefiles_test.pickle', 'rb') as handle:
    imagefiles = pickle.load(handle)


  labels_list = foi

  type1confusion = {}
  type2confusion = {}
  type2confusion2 = {}
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

  with open('imsitu_objects_directional_type2_confusion_test.csv', 'wb',0) as csvfile1:
    writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(["object1","object2", "test data type2 confusion"])
    

    for key, value in sorted(type2confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
      csvrecord = []
      csvrecord.append(key[0])
      csvrecord.append(key[1])
      csvrecord.append(value)
      writer.writerow(csvrecord)
      if (key[0],key[1]) not in type2confusion2:
        if (key[1],key[0]) in type2confusion:
          type2confusion2[key] = min(type2confusion[key], type2confusion[(key[1],key[0])])
        else:
          type2confusion2[key] = value

  with open('imsitu_objects_directional_type1_confusion_test.csv', 'wb',0) as csvfile1:
    writer = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(["object1","object2", "test data type1 confusion"])
    

    for key, value in sorted(type1confusion.iteritems(), key=lambda (k,v): (v,k), reverse=False):
      csvrecord = []
      csvrecord.append(key[0])
      csvrecord.append(key[1])
      csvrecord.append(value)
      writer.writerow(csvrecord)




if __name__ == "__main__":
  if len(sys.argv) == 1:
    dataset_path = "./imsitu/resized_256/" # change this path to dataset path
    #extract_test_data_coverage(dataset_path)
    get_test_data_yhats_labels_top100()
    deepinspect()
    generate_csv_test()
    output_two_types_confusion()
  else:    
    if len(sys.argv) != 2:
      print("python baseline_crf_deepinspect.py image")
      exit()
    predict_image_1(sys.argv[1])
