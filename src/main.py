import torch
import torch.nn.utils.prune as prune


import resnet
import convnext
import densenet
import vgg
import squeezenet
import mobilenetv2
import efficientnet
import inception

import time
import yaml
import sys
import os
import layers as l
import pruning as p
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomCrop, Normalize, Resize, InterpolationMode
from torchvision import models
from collections import OrderedDict


train_acc_vs_epoch = []
test_acc_vs_epoch = []

train_loss_vs_epoch = []
test_loss_vs_epoch = []



with open(os.getcwd() + "/parameters.yaml", "r") as yaml_file:
    all_parameters = yaml.safe_load(yaml_file)

epochs = all_parameters['epochs']
prune_start_epoch = all_parameters['prune_start_epoch']
batch_size = all_parameters['batch_size']
tau = all_parameters['tau']
# final_pruning = all_parameters['final_pruning']
final_pruning = float(sys.argv[-1])
eps_init = all_parameters['eps_init']
model_architecture = all_parameters['model_architecture']
pruning_type = all_parameters['pruning_type']
prune_schedule = all_parameters['prune_schedule']
test_type = all_parameters['test_type']
if pruning_type == 'regularized':
    regularization = True
else: 
    regularization = False

test_id = str(sys.argv[-2])


print(f"epochs: {epochs}")
print(f"batch_size: {batch_size}")
print(f"prune_start_epoch: {prune_start_epoch}")
print(f"tau: {tau}")
print(f"final_pruning: {final_pruning}")
print(f"eps_init: {eps_init}")
print(f"model_architecture: {model_architecture}")
print(f"regularization: {regularization}")
print(f"pruning_type: {pruning_type}")
print(f"prune_schedule: {prune_schedule}")


if model_architecture == "inception":

    upsample_transform_train = Compose([
        Resize(80, interpolation=InterpolationMode.NEAREST),
        RandomCrop(80, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    upsample_transform_test = Compose([
        Resize(80, interpolation=InterpolationMode.NEAREST),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

else:
    
    upsample_transform_train = Compose([
        Resize(64, interpolation=InterpolationMode.NEAREST),
        RandomCrop(64, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    upsample_transform_test = Compose([
        Resize(64, interpolation=InterpolationMode.NEAREST),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])






train_dataset = datasets.CIFAR10(
    root='data',
    train=True,
    transform=upsample_transform_train,  
    download=True
)



train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4  
)




test_dataset = datasets.CIFAR10(
    root='data',
    train=False,
    transform=upsample_transform_test,  
    download=True
)


test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4 
)




for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")





if model_architecture == 'resnet':

    model = resnet.resnet18(num_classes = 10, regularization=regularization)
    tf_model = models.resnet18(weights= 'DEFAULT')


elif model_architecture == 'vgg_bn' :

    model = vgg.vgg16_bn(regularization=regularization)
    model.classifier = nn.Sequential(
        nn.Flatten(),
        l.MyLinearLayer(512*7*7, 256, eps_init=0, regularization=regularization),
        nn.ReLU(),
        l.MyLinearLayer(256, 10, eps_init=0,regularization= regularization)
    )
    tf_model = models.vgg16_bn(weights= 'DEFAULT')

elif model_architecture == 'vgg' :

    model = vgg.vgg16(regularization=regularization)
    model.classifier = nn.Sequential(
        nn.Flatten(),
        l.MyLinearLayer(512*7*7, 256, eps_init=0.5,regularization=regularization),
        nn.ReLU(),
        l.MyLinearLayer(256, 10, eps_init=0.1,regularization=regularization)
    )
    tf_model = models.vgg16(weights= 'DEFAULT')

elif model_architecture == 'mobilenet':
    model = mobilenetv2.mobilenet_v2(num_classes = 10, regularization=regularization)
    tf_model = models.mobilenet_v2(weights ='DEFAULT')

elif model_architecture == 'efficientnet':
    model = efficientnet.efficientnet_v2_s(num_classes = 10, regularization=regularization)
    tf_model = models.efficientnet_v2_s(weights ='DEFAULT')

elif model_architecture == 'inception' :
    model = inception.inception_v3(num_classes = 10, regularization=regularization)
    tf_model = models.inception_v3(weights= 'DEFAULT')
    tf_model.AuxLogits = None


def get_conv_layers(model, pruning_list):
    for name, layer in model.named_children():
        if len(list(layer.children())) > 0:
            get_conv_layers(layer, pruning_list)
        else:
            if isinstance(layer, l.MyConvLayer) or isinstance(layer, l.MyConvLayerBN) or isinstance(layer, nn.Conv2d):
                pruning_list.append(layer)

tf_model_conv_layers = []
model_conv_layers = []

get_conv_layers(tf_model, tf_model_conv_layers)
get_conv_layers(model, model_conv_layers)


# Transfer Learning
for i in range(len(tf_model_conv_layers)-1):
    model_conv_layers[i].weight = tf_model_conv_layers[i].weight
    model_conv_layers[i].bias = tf_model_conv_layers[i].bias



print(model)

model = model.to(device)
loss_fn = nn.CrossEntropyLoss()

# High lr is better for models with BN 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def train(dataloader, model, loss_fn, optimizer, parameters_to_prune, epoch, final_pruning, tau, total_epochs, prune_start_epoch, pruning_type, schedule):
    size = len(dataloader.dataset)
    correct = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        step_per_epoch = size/(batch_size)
        if epoch >= prune_start_epoch:
            if schedule == 'exponential':
                percentage = p.prune_schedule_exp_decay(batch, epoch, prune_start_epoch, step_per_epoch, total_epochs, tau, final_pruning)
            elif schedule == 'one_shot':
                percentage = final_pruning
            elif schedule == 'prune_fnc':
                percentage = p.prune_schedule_polynomial_fnc(batch, epoch, prune_start_epoch, step_per_epoch, total_epochs, final_pruning)


            if pruning_type == 'regularized':
                p.prune_model(model, parameters_to_prune[0], parameters_to_prune[1], parameters_to_prune[2],percentage)
            else:
                if pruning_type == 'lamp':
                    p.prune_model_lamp(model, parameters_to_prune[0], parameters_to_prune[1], parameters_to_prune[2],percentage)
                elif pruning_type == 'global':
                    p.prune_model_global(model, parameters_to_prune[0], parameters_to_prune[1], parameters_to_prune[2],percentage)
                elif pruning_type == 'uniform':    
                    p.prune_model_uniform(model, parameters_to_prune[0], percentage)


        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Train Accuracy: {(100*(correct/current)):>0.1f}")
            if epoch >= prune_start_epoch:
                print(percentage)
    train_acc = (100*(correct/current))/batch_size
    train_loss = loss
    return train_acc, train_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    test_acc = 100*correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_acc, test_loss


parameters_to_prune = p.model_pruner_init(model)



scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

for t in range(epochs):
    start = time.time()
    print(f"Epoch {t+1}\n-------------------------------")
    tr_acc, tr_loss = train(train_dataloader, model, loss_fn, optimizer, parameters_to_prune, t+1, final_pruning, tau, epochs, prune_start_epoch, pruning_type, prune_schedule)
    ts_acc, ts_loss = test(test_dataloader, model, loss_fn)
    finish = time.time()
    print(f"Time Elapsed @Epoch {finish-start}")
    train_acc_vs_epoch.append(tr_acc)
    test_acc_vs_epoch.append(ts_acc)
    train_loss_vs_epoch.append(tr_loss)
    test_loss_vs_epoch.append(ts_loss)
    scheduler.step()

print("Done!")


result_dir = 'results/'+model_architecture+'/'


if test_type == 'pruning_schedule':
    result_dir = 'results/pruning_schedule/'+model_architecture+'/'+prune_schedule+'_'+pruning_type
elif test_type == 'pruning_type':
    result_dir = 'results/pruning_type/'+model_architecture+'/'+prune_schedule+'_'+pruning_type


with open((result_dir + '_' + test_id + '_train_acc_vs_epoch.txt'), 'w') as file:
    for e in train_acc_vs_epoch:
        file.write("%f\n" % e)
with open((result_dir + '_' + test_id + '_test_acc_vs_epoch.txt'), 'w') as file:
    for e in test_acc_vs_epoch:
        file.write("%f\n" % e)
with open((result_dir + '_' + test_id + '_train_loss_vs_epoch.txt'), 'w') as file:
    for e in train_loss_vs_epoch:
        file.write("%f\n" % e)
with open((result_dir + '_' + test_id + '_test_loss_vs_epoch.txt'), 'w') as file:
    for e in test_loss_vs_epoch:
        file.write("%f\n" % e)


model.eval()
        