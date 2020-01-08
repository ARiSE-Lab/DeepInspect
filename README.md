# DeepInspect: Testing DNN Image Classifier for Confusion & Bias Errors  (ICSE'20)
See the ICSE'20 paper [Testing DNN Image Classifier for Confusion & Bias Errors](https://arxiv.org/pdf/1905.07831.pdf) for more details.

There are generally two sections, "Reproduce paper results" section and "DeepInspect" section. The reproducing scipts in "Reproduce paper results" sections directly use the data our tool "DeepInspect" generated and then outputs the precision/recall, false postivie and true positive in our predictions. "DeepInspect" section includes the code on inspecting various datasets and models and generating the data for predictions.

## 1. Reproduce paper results

### Prerequisite
Python 3

### Reproducing results in Table 3 and Figure 6 in paper:  
```
cd reproduce
python3 confusion_bugs.py
```

### Reproducing results in Table 4 and Figure 10 in paper:
```
cd reproduce
python3 bias_bugs_estimate_ab_and_acd.py
python3 bias_bugs_generate_results.py
```
## 2. DeepInspect

### Inspect pre-trained COCO [model]()
#### Prerequisite
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, scikit-learn-0.20.4, matplotlib-2.2.4
#### Run deepinspect on pre-trained COCO model
```
cd deepinspect/coco/
python2 coco_deepinspect.py
```

### Inspect pre-trained COCO gender(COCO dataset with man/woman label) [model]()
#### Prerequisite
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, scikit-learn-0.20.4, matplotlib-2.2.4
#### Run deepinspect on pre-trained COCO gender model
```
cd deepinspect/coco_gender/
python2 coco_gender_deepinspect.py
```


### Inspect robust CIFAR-10 [models]()
#### Prerequisite
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, torchsummary

#### Run deepinspect on three robust training CIFAR-10 models
```
cd deepinspect/robust_cifar10/
python2 cifar10_small_deepinspect.py
python2 cifar10_large_deepinspect.py
python2 cifar10_resnet_deepinspect.py
```

### Inspect CIFAR-100 [model](https://github.com/aaron-xichen/pytorch-playground)
#### Prerequisite
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, jupyter

#### Run deepinspect on CIFAR-100 model
```
cd deepinspect/cifar100/
python2 cifar100_deepinspect.py
```
