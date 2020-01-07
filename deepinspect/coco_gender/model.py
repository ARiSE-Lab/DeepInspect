import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.utils

class MultilabelObject(nn.Module):

    def __init__(self, args, num_object):

        super(MultilabelObject, self).__init__()

        self.num_object = num_object

        self.base_network = models.resnet50(pretrained = True)
        if args is not None:
            if not args.finetune:
                for param in self.base_network.parameters():
                    param.requires_grad = False
        self.finalLayer = nn.Linear(self.base_network.fc.in_features, self.num_object)

    def forward(self, image):
        x = self.base_network.conv1(image)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)

        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)

        # avg pool or max pool
        x = self.base_network.avgpool(x)
        image_features = x.view(x.size(0), -1)
        x = self.finalLayer(image_features)
        return x