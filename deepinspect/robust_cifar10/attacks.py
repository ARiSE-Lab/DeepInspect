import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import problems as pblm
import argparse
import numpy as np
import random


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

def mean(l): 
	return sum(l)/len(l)

def _fgs(model, X, y, epsilon): 
	opt = optim.Adam([X], lr=1e-3)
	out = model(X)
	ce = nn.CrossEntropyLoss()(out, y)
	err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

	opt.zero_grad()
	ce.backward()
	eta = X.grad.data.sign()*epsilon
	
	X_fgs = Variable(X.data + eta)
	err_fgs = (model(X_fgs).data.max(1)[1] != y.data).float().sum()  / X.size(0)
	return err, err_fgs

def fgs(loader, model, epsilon, verbose=False, robust=False): 
	return attack(loader, model, epsilon, verbose=verbose, atk=_fgs,
				  robust=robust)


def _pgd(model, X, y, epsilon, niters=40, alpha=0.01): 
	out = model(X)
	ce = nn.CrossEntropyLoss()(out, y)
	err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

	X_pgd = Variable(X.data, requires_grad=True)
	for i in range(niters): 
		opt = optim.Adam([X_pgd], lr=1e-3)
		opt.zero_grad()
		loss = nn.CrossEntropyLoss()(model(X_pgd), y)
		loss.backward()
		eta = alpha*X_pgd.grad.data.sign()
		X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
		
		# adjust to be within [-epsilon, epsilon]
		eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
		#X_pgd = Variable(torch.clamp(X.data + eta, 0.0, 1.0), requires_grad=True)
		X_pgd = Variable(X.data + eta, requires_grad=True)
		
	err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
	return err, err_pgd


def _cw(model, X, y, epsilon, niters=40, alpha=0.01, X_nat=None): 
	if(X_nat is None):
		X_nat = Variable(X.data)
	out = model(X)
	ce = nn.CrossEntropyLoss()(out, y)
	err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

	X_cw = Variable(X.data, requires_grad=True)
	for i in range(niters): 
		opt = optim.Adam([X_cw], lr=1e-3)
		opt.zero_grad()

		pre_softmax = model(X_cw)

		if(use_cuda):
			y_onehot = torch.cuda.FloatTensor(y.shape[0], 10)
		else:
			y_onehot = torch.FloatTensor(y.shape[0], 10)
		y_onehot.zero_()

		y_onehot.scatter_(1, y.unsqueeze(-1), 1)

		real = (y_onehot * pre_softmax).sum(1)
		other = ((1. - y_onehot) * pre_softmax - y_onehot * 10000.).max(1)[0]

		loss = torch.sum(-torch.clamp(real - other + 50, min=0.))
		loss.backward()

		eta = alpha*X_cw.grad.data.sign()
		X_cw = Variable(X_cw.data + eta, requires_grad=True)
		
		# adjust to be within [-epsilon, epsilon]
		eta = torch.clamp(X_cw.data - X_nat.data, -epsilon, epsilon)

		#X_cw = Variable(torch.clamp(X_nat.data + eta, 0.0, 1.0), requires_grad=True)
		X_cw = Variable(X_nat.data + eta, requires_grad=True)
		#X_cw = Variable(X_cw.data, requires_grad=True)
		
	err_cw = (model(X_cw).data.max(1)[1] != y.data).float().sum() / X.size(0)
	return err, err_cw, X_cw

def reduce_sum(x, keepdim=True):
	# silly PyTorch, when will you get proper reducing sums/means?
	for a in reversed(range(1, x.dim())):
		x = x.sum(a, keepdim=keepdim)
	return x



def check_is_adv(array, model, y):
	if(use_cuda):
		array = Variable(torch.FloatTensor(array).cuda())
	else:
		array = Variable(torch.FloatTensor(array))
	return (model(array).data.max(1)[1] != y.data)


def l0_norm(x1,x2):
	return np.sum(((x1-x2)!=0).astype(np.int32))



def pgd(loader, model, epsilon, niters=100, alpha=0.01, verbose=False,
		robust=False):
	return attack(loader, model, epsilon, verbose=verbose, atk=_pgd,
				  robust=robust)

def attack(loader, model, epsilon, verbose=False, atk=None,
		   robust=False):
	
	total_err, total_fgs, total_robust = [],[],[]
	if verbose: 
		print("Requiring no gradients for parameters.")
	for p in model.parameters(): 
		p.requires_grad = False
	
	for i, (X,y) in enumerate(loader):
		X,y = Variable(X.cuda(), requires_grad=True), Variable(y.cuda().long())

		if y.dim() == 2: 
			y = y.squeeze(1)
		
		if robust: 
			robust_ce, robust_err = robust_loss_batch(model, epsilon, X, y, False, False)

		err, err_fgs = atk(model, X, y, epsilon)
		
		total_err.append(err)
		total_fgs.append(err_fgs)
		if robust: 
			total_robust.append(robust_err)
		if verbose: 
			if robust: 
				print('err: {} | attack: {} | robust: {}'.format(err, err_fgs, robust_err))
			else:
				print('err: {} | attack: {}'.format(err, err_fgs))
	
	if robust:		 
		print('[TOTAL] err: {} | attack: {} | robust: {}'.format(mean(total_err), mean(total_fgs), mean(total_robust)))
	else:
		print('[TOTAL] err: {} | attack: {}'.format(mean(total_err), mean(total_fgs)))
	return total_err, total_fgs, total_robust



torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

use_cuda = torch.cuda.is_available()

