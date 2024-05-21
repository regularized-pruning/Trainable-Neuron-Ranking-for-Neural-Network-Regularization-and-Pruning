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


#############################################################################
def get_modules(model):
    lyrs, _, _ = model_pruner_init(model)
    return lyrs

def get_weights(model):
    _, ws, _ = model_pruner_init(model)
    w = []
    for i in ws:
        w.append(i[0].weight)
    return w

def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(((m.weight != 0)*1).sum())
    return torch.FloatTensor(unmaskeds)


def _normalize_scores(scores):
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

def _compute_lamp_amounts(model,amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum()*(1.0-amount)))
    
    flattened_scores = [_normalize_scores(w[0].weight**2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores,dim=0)
    topks,_ = torch.topk(concat_scores,num_surv)
    threshold = topks[-1]
    
    # We don't care much about tiebreakers, for now.
    final_survs = [torch.ge(score,threshold*torch.ones(score.size()).to(score.device)).sum() for score in flattened_scores]
    amounts = []
    for idx,final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv/unmaskeds[idx]))
    
    return amounts

def prune_weights_l1predefined(model,amounts):
    mlist = get_modules(model)
    for idx,m in enumerate(mlist):
        prune.l1_unstructured(m,name="weight",amount=float(amounts[idx]))

def prune_weights_lamp(model,amount):
    assert amount <= 1
    amounts = _compute_lamp_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

######################################################################################################################
#TODO:CHECK THIS AND FIX

def _amounts_from_eps(unmaskeds,ers,amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0-amount)*unmaskeds.sum() # Total to keep.
    
    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds*(1-layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense*unmaskeds).sum()
        
        ers_of_prunables = ers*(1.0-layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables*ers_of_prunables/ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx]/unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx]/unmaskeds[idx]
        
        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)
    
    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx]/unmaskeds[idx])
    return amounts

def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx,w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0)+w.size(1)+w.size(2)+w.size(3)
        else:
            erks[idx] = w.size(0)+w.size(1)
    return erks

def _compute_erk_amounts(model,amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)

    return _amounts_from_eps(unmaskeds,erks,amount)


def prune_weights_erk(model,amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

##########################################################################
    
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