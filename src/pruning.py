#LAMP Score is taken from: https://github.com/jaeho-lee/layer-adaptive-sparsity


import torch
import layers as l
import torch.nn.utils.prune as prune
from torch import nn
from collections import OrderedDict

import numpy as np


def lamp_normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
    new_scores[sorted_idx] = sorted_scores
    
    return new_scores.view(scores.shape)


def get_pruning_list(model, pruning_list):
    for name, layer in model.named_children():
        if len(list(layer.children())) > 0:
            get_pruning_list(layer, pruning_list)
        else:
            if isinstance(layer, l.MyLinearLayer) or isinstance(layer, l.MyConvLayer) or isinstance(layer, l.MyConvLayerBN):
                pruning_list.append(layer)


def get_pruning_list_base(model, pruning_list):
    for name, layer in model.named_children():
        if len(list(layer.children())) > 0:
            get_pruning_list_base(layer, pruning_list)
        else:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                pruning_list.append(layer)



def model_pruner_init(model):

    layer_list = []
    parameters_to_prune_w = []
    parameters_to_prune_b = []

    get_pruning_list(model,layer_list)

    for lyr in layer_list:
        parameters_to_prune_w.append((lyr, 'weight'))
        if lyr.bias != None:
            parameters_to_prune_b.append((lyr, 'bias'))

    parameters_to_prune_w = tuple(parameters_to_prune_w)
    parameters_to_prune_b = tuple(parameters_to_prune_b)

    return layer_list, parameters_to_prune_w, parameters_to_prune_b


def model_pruner_init_base(model):

    layer_list = []
    parameters_to_prune_w = []
    parameters_to_prune_b = []

    get_pruning_list_base(model,layer_list)

    for lyr in layer_list:
        parameters_to_prune_w.append((lyr, 'weight'))
        if lyr.bias != None:
            parameters_to_prune_b.append((lyr, 'bias'))

    parameters_to_prune_w = tuple(parameters_to_prune_w)
    parameters_to_prune_b = tuple(parameters_to_prune_b)

    return layer_list, parameters_to_prune_w, parameters_to_prune_b

def prune_model(model, layer_list, parameters_to_prune_w, parameters_to_prune_b, percentage):

    importance_dict_w = OrderedDict()
    importance_dict_b = OrderedDict()

    for i, lyr in enumerate(layer_list):
        if isinstance(lyr, l.MyConvLayer):
            s = lyr.state_dict()['ch_score'][:,0,0]
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = s * torch.abs(lyr.bias)
                importance_dict_b[bias_tuple] = lamp_normalize_scores(importance_dict_b[bias_tuple])

            s = s[:,None, None, None]
            importance_dict_w[weight_tuple] = s * torch.abs(lyr.weight)
            importance_dict_w[weight_tuple] = lamp_normalize_scores(importance_dict_w[weight_tuple])

        if isinstance(lyr, l.MyConvLayerBN):
            s = lyr.state_dict()['ch_score'][:,0,0]
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = s * torch.abs(lyr.bias)
                importance_dict_b[bias_tuple] = lamp_normalize_scores(importance_dict_b[bias_tuple])

            s = s[:,None, None, None]
            importance_dict_w[weight_tuple] = s * torch.abs(lyr.weight)
            importance_dict_w[weight_tuple] = lamp_normalize_scores(importance_dict_w[weight_tuple])

        if isinstance(lyr, l.MyLinearLayer):
            s = lyr.state_dict()['ch_score']
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = s * torch.abs(lyr.bias)
                importance_dict_b[bias_tuple] = lamp_normalize_scores(importance_dict_b[bias_tuple])

            s = s[:,None]
            importance_dict_w[weight_tuple] = s * torch.abs(lyr.weight)
            importance_dict_w[weight_tuple] = lamp_normalize_scores(importance_dict_w[weight_tuple])


    #Prune Weight
    prune.global_unstructured(parameters_to_prune_w, pruning_method=prune.L1Unstructured,amount=percentage, importance_scores=importance_dict_w)
    #Prune Bias
    prune.global_unstructured(parameters_to_prune_b, pruning_method=prune.L1Unstructured,amount=percentage, importance_scores=importance_dict_b)
    
    for i, lyr in enumerate(layer_list):
        if isinstance(lyr, l.MyConvLayer):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')
        if isinstance(lyr, l.MyConvLayerBN):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')
        if isinstance(lyr, l.MyLinearLayer):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')


                

def prune_model_global(model, layer_list, parameters_to_prune_w, parameters_to_prune_b, percentage):

    importance_dict_w = OrderedDict()
    importance_dict_b = OrderedDict()

    for i, lyr in enumerate(layer_list):
        if isinstance(lyr, l.MyConvLayer):
            s = lyr.state_dict()['ch_score'][:,0,0]
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = torch.abs(lyr.bias)

            s = s[:,None, None, None]
            importance_dict_w[weight_tuple] = torch.abs(lyr.weight)


        if isinstance(lyr, l.MyConvLayerBN):
            s = lyr.state_dict()['ch_score'][:,0,0]
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = torch.abs(lyr.bias)
           
            s = s[:,None, None, None]
            importance_dict_w[weight_tuple] = torch.abs(lyr.weight)
          
        if isinstance(lyr, l.MyLinearLayer):
            s = lyr.state_dict()['ch_score']
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = torch.abs(lyr.bias)

            s = s[:,None]
            importance_dict_w[weight_tuple] = torch.abs(lyr.weight)
          


    #Prune Weight
    prune.global_unstructured(parameters_to_prune_w, pruning_method=prune.L1Unstructured,amount=percentage, importance_scores=importance_dict_w)
    #Prune Bias
    prune.global_unstructured(parameters_to_prune_b, pruning_method=prune.L1Unstructured,amount=percentage, importance_scores=importance_dict_b)


    for i, lyr in enumerate(layer_list):
        if isinstance(lyr, l.MyConvLayer):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')
        if isinstance(lyr, l.MyConvLayerBN):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')
        if isinstance(lyr, l.MyLinearLayer):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')

def prune_model_lamp(model, layer_list, parameters_to_prune_w, parameters_to_prune_b, percentage):

    importance_dict_w = OrderedDict()
    importance_dict_b = OrderedDict()

    
    for i, lyr in enumerate(layer_list):
        if isinstance(lyr, l.MyConvLayer):
            s = lyr.state_dict()['ch_score'][:,0,0]
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = torch.abs(lyr.bias)
                importance_dict_b[bias_tuple] = lamp_normalize_scores(importance_dict_b[bias_tuple])

            s = s[:,None, None, None]
            importance_dict_w[weight_tuple] = torch.abs(lyr.weight)
            importance_dict_w[weight_tuple] = lamp_normalize_scores(importance_dict_w[weight_tuple])

        if isinstance(lyr, l.MyConvLayerBN):
            s = lyr.state_dict()['ch_score'][:,0,0]
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = torch.abs(lyr.bias)
                importance_dict_b[bias_tuple] = lamp_normalize_scores(importance_dict_b[bias_tuple])

            s = s[:,None, None, None]
            importance_dict_w[weight_tuple] = torch.abs(lyr.weight)
            importance_dict_w[weight_tuple] = lamp_normalize_scores(importance_dict_w[weight_tuple])

        if isinstance(lyr, l.MyLinearLayer):
            s = lyr.state_dict()['ch_score']
            weight_tuple = (lyr, 'weight')
            bias_tuple = (lyr, 'bias')

            if lyr.bias != None:
                importance_dict_b[bias_tuple] = torch.abs(lyr.bias)
                importance_dict_b[bias_tuple] = lamp_normalize_scores(importance_dict_b[bias_tuple])

            s = s[:,None]
            importance_dict_w[weight_tuple] = torch.abs(lyr.weight)
            importance_dict_w[weight_tuple] = lamp_normalize_scores(importance_dict_w[weight_tuple])



    #Prune Weight
    prune.global_unstructured(parameters_to_prune_w, pruning_method=prune.L1Unstructured,amount=percentage, importance_scores=importance_dict_w)
    #Prune Bias
    prune.global_unstructured(parameters_to_prune_b, pruning_method=prune.L1Unstructured,amount=percentage, importance_scores=importance_dict_b)


    for i, lyr in enumerate(layer_list):
        if isinstance(lyr, l.MyConvLayer):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')
        if isinstance(lyr, l.MyConvLayerBN):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')
        if isinstance(lyr, l.MyLinearLayer):
            prune.remove(lyr, 'weight')
            if lyr.bias != None:
                prune.remove(lyr, 'bias')


    
def prune_model_uniform(model, layer_list, percentage):
    assert percentage <= 1 
    for lyr in layer_list:
        prune.l1_unstructured(lyr,name="weight",amount=percentage)
        prune.remove(lyr, 'weight')
        if lyr.bias != None:
            prune.l1_unstructured(lyr,name="bias",amount=percentage)
            prune.remove(lyr, 'bias')
##########################################################################
            
def prune_schedule_exp_decay(batch, epoch, prune_start_epoch, step_per_epoch, total_epochs, tau, final_pruning):
    percentage = (batch + ((epoch-prune_start_epoch)*step_per_epoch))/((total_epochs-prune_start_epoch)*step_per_epoch)
    percentage = 1 - torch.exp(-tau*torch.tensor(percentage)).item()
    percentage = final_pruning*percentage

    return percentage

def prune_schedule_polynomial_fnc(batch, epoch, prune_start_epoch, step_per_epoch, total_epochs, final_pruning):
    percentage = (batch + ((epoch-prune_start_epoch)*step_per_epoch))/((total_epochs-prune_start_epoch)*step_per_epoch)
    percentage = 1 - (1 - percentage)**3
    percentage = final_pruning*percentage
    return percentage
