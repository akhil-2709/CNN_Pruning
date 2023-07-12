# mount google dive which has all the models which we trained
# from google.colab import drive
# drive.mount("/content/drive", force_remount=True)

# import sys
# sys.path.append("/content/drive/My Drive/cs532_project2")

#importing all the libraries needed to run the test script

import warnings

warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# defining the convolution where dimensions are 3x3
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


# defining the convolution where dimensions are 1x1
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#creating a class for the basic block
class BasicBlock(nn.Module):
    expansion = 1

#initializing the attribute for the basic block class/
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
#check if layer is none
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

#checking if groups not equal to 1 
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#check if dilation greater than 1 
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
# set attributes for the basicblock 
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

# function which carries out the forward pass 
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#class for the bottleneck of the cnn
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#class for ResNet 18 network
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load("./resnet18.pt")
        # print(type(state_dict))
        # print(state_dict)
        # state_dict = torch.load(
        #     script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        # )
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )


from torchvision import transforms as T

#Mean and standard deviation of the data calculated in order to normalize the data
mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)

transform = T.Compose([T.ToTensor(), T.Normalize(mean, std),])

# Obtain the train and test datasets from CIFAR-10 dataset

test_dataset = torchvision.datasets.CIFAR10(
    root="../../data/", train=False, transform=transform, download=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=True
)


import time as time

#Configuration of the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### function for calculating the accuracy and other metrics for the model 
def model_stats(model):
    num_correct = 0
    num_samples = 0
    model.eval()
    start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        end = time.time()
        accuracy = float(num_correct) / num_samples
        inference_time_per_image = (end - start) / num_samples

    return accuracy, inference_time_per_image, num_samples


model = resnet18(pretrained=True)

model.load_state_dict(torch.load("./resnet18.pt", map_location=torch.device("cpu"),))
model.to(device)
print("ResNet-18 stats before pruning on the test images\n")
start = time.time()
accuracy_before_prune, inference_time_per_img_before_prune, num_samples = model_stats(
    model
)
end = time.time()
inference_time_per_img_before_prune = (end - start) / num_samples

print("Accuracy of the model: {:.4f}%".format(accuracy_before_prune * 100))
print(
    "Inference time per image: {:.7f} seconds".format(
        inference_time_per_img_before_prune
    )
)
print("-----------------------------------------")


### Storing all the model names
models = [
    "resnet18_prune_50perc_oneshot3.pt",
    "resnet18_prune_50perc_oneshot5.pt",
    "resnet18_prune_75perc_oneshot3.pt",
    "resnet18_prune_75perc_oneshot5.pt",
    "resnet18_prune_90perc_oneshot3.pt",
    "resnet18_prune_90perc_oneshot5.pt",
    "resnet18_prune_50perc_iterative3_3.pt",
    "resnet18_prune_50perc_iterative5_5.pt",
    "resnet18_prune_75perc_iterative3_3.pt",
    "resnet18_prune_75perc_iterative5_5.pt",
    "resnet18_prune_90perc_iterative3_3.pt",
    "resnet18_prune_90perc_iterative5_5.pt",
]

# for all the models stored calculating the accuracies on the train and test sets
for model_id in models:
    model = resnet18(pretrained=True)
    model.load_state_dict(
        torch.load("./" + str(model_id), map_location=torch.device("cpu"),)
    )
    model.to(device)
    print("Pruned Model Stats =>")
    print("Model Name: ", model_id)
    start = time.time()
    accuracy, inference_time_per_img, num_samples = model_stats(model)
    end = time.time()
    inference_time_per_img = (end - start) / num_samples
    print("Accuracy of the pruned model: {:.4f}%".format(accuracy * 100))
    print("Accuracy Drop: {:.4f}%".format(100.0 * (accuracy_before_prune - accuracy)))
    print(
        "Inference time per image before pruning: {:.7f} seconds".format(
            inference_time_per_img_before_prune
        )
    )
    print(
        "Inference time per image after pruning: {:.7f} seconds".format(
            inference_time_per_img
        )
    )
    print(
        "Speedup: {:.7f}".format(
            inference_time_per_img_before_prune / inference_time_per_img
        )
    )
    print("--------------------------------------------------------------")
