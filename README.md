# DeepInspect: Testing DNN Image Classifier for Confusion & Bias Errors  (ICSE'20)
See the ICSE'20 paper [Testing DNN Image Classifier for Confusion & Bias Errors](https://arxiv.org/pdf/1905.07831.pdf) for more details.


## Reproduce paper results

### Prerequisite
Python 3

### Reproducing results in Table 3 and Figure 6 in paper:  
```
cd reproduce
python3 confusion_bugs.py
```
## DeepInspect

### Inspect robust CIFAR-10 [models]()
#### Prerequisite
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, torchsummary

#### Run deepinspect on cifar-100 model
```
cd deepinspect/robust_cifar10/
python2 cifar10_small_deepinspect.py
python2 cifar10_large_deepinspect.py
python2 cifar10_resnet_deepinspect.py
```

### Inspect CIFAR-100 [model](https://github.com/aaron-xichen/pytorch-playground)
#### Prerequisite
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, jupyter

#### Run deepinspect on cifar-100 model
```
cd deepinspect/cifar100/
python2 cifar100_deepinspect.py
```
