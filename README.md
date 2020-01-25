# DeepInspect: Testing DNN Image Classifier for Confusion & Bias Errors  (ICSE'20)
See the ICSE'20 paper [Testing DNN Image Classifier for Confusion & Bias Errors](https://arxiv.org/pdf/1905.07831.pdf) for more details.

There are two directories: (1) Reproduce paper results and (2) DeepInspect. **DeepInspect** is the implementation of the tool that analyzes the target model and dataset under test and outputs potential confusion and bias errors. The scripts in **Reproduce paper results** directory evaluate DeepInspect, i.e., they analyze the tool's output and report precision/recall, false positive and true positive of our predictions. 

## 1. Reproduce paper results

### Prerequisite
```
Python 3, numpy, scipy, matplotlib, sklearn, pandas
```
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
Generating neuron coverage and computing probability matrix for large dataset may take hours. It is recommended to run the CIFAR-10 script first to get a general idea of DeepInspect, set up the environment and get familiar with the whole workflow. 

Note1: neuron coverage for each dataset are only required to compute once, so you can comment out the function call(with name like *get_coverage*) in the main function in each script. The neuron coverage is saved into a pickle file with name like "globalcoverage*.pickle". Since it is cumulatively saved to this pickle file, if you need to re-call the get_coverage function again, please remove this pickle file before calling it again.

Note2: we only use 90% of test data's predictions to predict confusion and bias bugs, that is why we use sample_10 as parameter in scripts to intentionally keep 10% of data for other use in future. You can always set sample_10 to empty list to leverage all the test data's predictions.

### 2.1. Inspect pre-trained COCO model (model from [paper](https://arxiv.org/abs/1707.09457))
#### Prerequisite
```
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, scikit-learn-0.20.4, matplotlib-2.2.4
```
#### Download [COCO 2014 dataset](http://cocodataset.org/#download)

#### COCO dataset structure
```
cocodataset  
├── annotations  
│   ├── instances_train2014.json             
│   ├── instances_val2014.json  
├── train2014                    
│   ├── COCO_train2014_000000291797.jpg      
│   ├── ...     
├── val2014                   
│   ├── COCO_val2014_000000581929.jpg               
│   ├── ...                    
```
#### Run deepinspect on pre-trained COCO model
```
cd deepinspect/coco/
python2 coco_deepinspect.py
```
```
Copy the generated csv files to override the files in data/coco/ folder and run the code in Section 1.
```


### 2.2. Inspect pre-trained COCO gender(COCO dataset with man/woman label) model (model from [paper](https://arxiv.org/abs/1707.09457))
#### Prerequisite
```
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, scikit-learn-0.20.4, matplotlib-2.2.4
```
#### COCO gender dataset structure
```
Same as COCO dataset structure
```
#### Run deepinspect on pre-trained COCO gender model
```
cd deepinspect/coco_gender/
python2 coco_gender_deepinspect.py
```
```
Copy the generated csv files to override the files in data/coco_gender/ folder and run the code in Section 1.
```

### 2.3. Inspect robust CIFAR-10 models (models from [paper1](http://papers.nips.cc/paper/8060-scaling-provable-adversarial-defenses.pdf), [paper2](https://arxiv.org/abs/1811.02625))
#### Prerequisite
```
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, torchsummary
```
#### Run deepinspect on three robust training CIFAR-10 models
```
cd deepinspect/robust_cifar10/
python2 cifar10_small_deepinspect.py
python2 cifar10_large_deepinspect.py
python2 cifar10_resnet_deepinspect.py
```
```
Copy the generated csv files to override the files in data/robust_cifar10_{small/large/resnet/}/ folder and run the code in Section 1.
```

### 2.4. Inspect CIFAR-100 model (model from [repo](https://github.com/aaron-xichen/pytorch-playground))
#### Prerequisite
```
Python 2.7, numpy-1.16, tqdm-4.41, torch-1.3.1, jupyter
```
#### Run deepinspect on CIFAR-100 model
```
cd deepinspect/cifar100/
python2 cifar100_deepinspect.py
```
```
Copy the generated csv files to override the files in data/cifar100/ folder and run the code in Section 1.
```

### 2.5. Inspect pre-trained ImageNet model for ILSVRC2012 dataset(model from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html))
#### Prerequisite
```
Python 2.7, numpy-1.16, tqdm-4.41, torch, opencv-python, torchvision, cPickle, Pillow
```
#### Run deepinspect on ImageNet model
```
cd deepinspect/imagenet/
python2 imagenet_coverage.py
python2 Imagenet_deepinspect.py
```
```
Copy the generated csv files to override the files in data/imagenet/ folder and run the code in Section 1.
```


### 2.6. Inspect pre-trained baseline_crf ResNet model for imSitu dataset(model from [paper](https://github.com/my89/imSitu))
#### Prerequisite
```
Python 2.7, numpy-1.16, tqdm-4.41, torch **0.3.1**, opencv-python, torchvision **0.2**, pickle
```
#### Run deepinspect on imSitu model

Install the original models from [imSitu](https://github.com/my89/imSitu)  
Copy folder *baseline_models* to *deepinspect/imsitu/*  
Download dataset to folder *resized_256/*.
```
cd deepinspect/imsitu/
python2 baseline_crf_deepinspect.py resized_256/adjusting_1.jpg # test environment, this should output predictions without any error.

python2 baseline_crf_deepinspect.py
```
```
Copy the generated csv files to override the files in data/imsitu/ folder and run the code in Section 1.
```

