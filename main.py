import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vog_calc(trainloader, model,dataset_name):
    vog = {}
    for idx, batch in enumerate(trainloader):
        if idx >= 10:
            print(idx)
            break
        X,y = batch
        X, y = X.to(device), y.to(device)
        grad_X = X
        grad_Y = y
        grad_X.requires_grad = True
        logits = model(grad_X)
        celoss = nn.CrossEntropyLoss()
        loss = celoss(logits, grad_Y)
        loss.backward()
        sumMatrix = torch.sum(grad_X.grad, dim=0)
        
    sumMatrix /= grad_X.grad.shape[0] * len(trainloader)
    covMatrix = [torch.pow(covMatric - sumMatrix, 2) for covMatric in grad_X]
    covMatrix = sum(covMatrix)
    covMatrix /= len(trainloader)
    meanMatrix = torch.mean(covMatrix)
    print("Dataset_ {}: {}".format(dataset_name, meanMatrix.item()))
    
    # print(covGrad)    


import torch.nn as nn
import torchvision
class MyResNet(nn.Module):

    def __init__(self, in_channels=1):
        super(MyResNet, self).__init__()

        # bring resnet
        self.model = torchvision.models.resnet18(pretrained=True)

        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # your case
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)

model = MyResNet()
model_rgb = torchvision.models.resnet18(pretrained=True)
import torch


from torchvision import transforms, datasets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
     
trainset = datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

model.to(device)
model = nn.DataParallel(model)
model.eval()


model_rgb.to(device)
model_rgb = nn.DataParallel(model_rgb)
model_rgb.eval()

vog = vog_calc(trainloader, model, "MNIST")

from torchvision import transforms, datasets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
     

trainset = datasets.KMNIST(root='./data2', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
'''
testset = datasets.SEMEION(root='./data2', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

'''
vog = vog_calc(trainloader, model, "KMNIST")

from torchvision import transforms, datasets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
     

trainset = datasets.FashionMNIST(root='./data2', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
'''
testset = datasets.SEMEION(root='./data2', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

'''
vog = vog_calc(trainloader, model, "FashionMNIST")
print(vog)


from torchvision import transforms, datasets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
     

trainset = datasets.CIFAR10(root='./data2', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
'''
testset = datasets.SEMEION(root='./data2', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

'''
vog = vog_calc(trainloader, model_rgb, "CIFAR10")
print(vog)