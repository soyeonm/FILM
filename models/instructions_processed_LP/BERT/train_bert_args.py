#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:49:31 2021

@author: soyeonmin
"""

import random
import time
import torch
from torch import nn
import os
import glob
from collections import OrderedDict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-lr','--learning_rate', type=float, default=1e-5, help="learning rate")
parser.add_argument('-s','--seed', type=int, default=0, help="seed")
parser.add_argument('-l','--label', type=int, default=1, help="template")
parser.add_argument('-d','--decay', type=float, default=0.5, help="template")
parser.add_argument('-dt','--decay_term', type=int, default=5, help="template")
parser.add_argument('-t','--task', type=str, help="type in one of mrecep, object, parent, toggle, sliced")
parser.add_argument('-v','--verbose', type=int, default=0, help="print training output")
parser.add_argument('-model_type','--model_type', type=str, default='bert-base', help="one of roberta-large, roberta-base, bert-base, bert-large")
parser.add_argument('-load','--load', type=str, default='', help="one of roberta-large, roberta-base, bert-base, bert-large")
parser.add_argument('-no_divided_label','--no_divided_label', action='store_true')
parser.add_argument('--no_appended', action='store_true')


args = parser.parse_args()

#Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
import pickle
template_by_label = pickle.load(open('data/alfred_data/alfred_dicts/template_by_label.p', 'rb'))
train_set = pickle.load(open('data/alfred_data/train_text_with_ppdl_low_appended.p', 'rb'))
val_set_seen = pickle.load(open('data/alfred_data/val_seen_text_with_ppdl_low_appended.p', 'rb'))
val_set_unseen = pickle.load(open('data/alfred_data/val_unseen_text_with_ppdl_low_appended.p', 'rb'))

obj2idx = pickle.load(open('data/alfred_data/alfred_dicts/obj2idx.p', 'rb'))
recep2idx = pickle.load(open('data/alfred_data/alfred_dicts/recep2idx.p', 'rb'))
toggle2idx = pickle.load(open('data/alfred_data/alfred_dicts/toggle2idx.p', 'rb'))

assert args.task in ['mrecep', 'object', 'parent', 'toggle', 'sliced']
if args.task == 'mrecep':
    label_arg = 'mrecep_targets'; num_labels = len(obj2idx)
elif args.task == 'object':
    label_arg = 'object_targets'; num_labels = len(obj2idx)
elif args.task == 'parent':
    label_arg = 'parent_targets'; num_labels = len(recep2idx)
elif args.task == 'toggle':
    label_arg = 'toggle_targets'; num_labels = len(toggle2idx)
elif args.task == 'sliced':
    label_arg = 's'; num_labels = 2





if args.model_type in ['bert-base', 'bert-large'] : 
    from transformers import BertTokenizer as Tokenizer
    tok_type = args.model_type + '-uncased'
    from transformers import BertForSequenceClassification as BertModel
elif args.model_type in ['roberta-base', 'roberta-large']:
    from transformers import RobertaTokenizer as Tokenizer
    tok_type = args.model_type
    from transformers import RobertaForSequenceClassification as BertModel
tokenizer = Tokenizer.from_pretrained(tok_type)

torch.manual_seed(args.seed)
model = BertModel.from_pretrained(tok_type, num_labels=num_labels)

if len(args.load) >0:
    sd = OrderedDict()
    sd_ori = torch.load(args.load, map_location = 'cpu')
    for k in sd_ori:
        if 'classifier' in k:
            if k =='classifier.weight':
                sd[k] = model.classifier.weight
            elif k =='classifier.bias':
                sd[k] = model.classifier.bias
        else:
            sd[k] = sd_ori[k]
    del sd_ori
    
    model.load_state_dict(sd)

model = model.to(device)

####################################################
## 1. Prepare Data
####################################################


from random import shuffle
import numpy as np
x_train = train_set['x_low']; y_train = train_set['y']
if args.no_appended:
    x_train = train_set['x']
#if args.no_divided_label:
x_train = [str(y_train[i])+ ' ' + x for i, x in enumerate(x_train)]

if args.task == 'sliced' or args.no_divided_label:
    x_ones = [i for i in range(len(x_train))]
else:
    x_ones = [i for i, lab in enumerate(y_train) if lab == args.label]
random.seed(0)
shuffled = [i for i in x_ones]
shuffle(shuffled)
#x_train = np.array(train_set['x_low'])[shuffled]
x_train = np.array(x_train)[shuffled]
x_train = x_train.tolist()
encoding = tokenizer(x_train, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids'].to(device) #has shape [21025, 37]
attention_mask = encoding['attention_mask'].to(device) #also has shape [21025, 37]

label_train = np.array(train_set[label_arg])[shuffled]
label_train = torch.tensor(label_train).to(device)



x_val_seen = val_set_seen['x_low']; y_val_seen = val_set_seen['y']
if args.no_appended:
    x_val_seen = val_set_seen['x']
x_val_seen = [str(y_val_seen[i])+ ' ' + x for i, x in enumerate(x_val_seen)]

if args.task == 'sliced' or args.no_divided_label:
    random.seed(0)
    x_ones_vs = random.sample(range(len(x_val_seen)), 100)
else:
    x_ones_vs = [i for i, lab in enumerate(y_val_seen) if lab == args.label]
x_val_seen = np.array(x_val_seen)[x_ones_vs].tolist()
encoding_v_s = tokenizer(x_val_seen, return_tensors='pt', padding=True, truncation=True)
input_ids_val_seen = encoding_v_s['input_ids'].to(device)
attention_mask_val_seen = encoding_v_s['attention_mask'].to(device) 

label_val_seen = [val_set_seen[label_arg][i] for i in x_ones_vs]
label_val_seen = torch.tensor(label_val_seen).to(device)



####################################################
## 2. Do training
####################################################

model.train()
from transformers import AdamW
learning_rate = args.learning_rate
optimizer = AdamW(model.parameters(), lr=learning_rate)
if 'base' in args.model_type:
    N= 64
else:
    N = 32

def accuracy(y_pred, y_batch):
    #y_pred has shape [batch, no_classes]
    maxed = torch.max(y_pred, 1)
    y_hat = maxed.indices
    num_accurate = torch.sum((y_hat == y_batch).float())
    train_accuracy = num_accurate/ y_hat.shape[0]
    return train_accuracy.item()

def accurate_total(y_pred, y_batch):
    #y_pred has shape [batch, no_classes]
    maxed = torch.max(y_pred, 1)
    y_hat = maxed.indices
    num_accurate = torch.sum((y_hat == y_batch).float())
    return num_accurate


if args.no_appended:
    super_folder = 'saved_models_noappended/'
else:
    super_folder = 'saved_models/'

if args.task == 'sliced':
    save_folder_name = args.task +'/' + args.model_type + '_lr_' + str(args.learning_rate) + 'seed_' + str(args.seed) + 'decay_' + str(args.decay) +'/'
else:
    save_folder_name = args.task + '/template_' + str(args.label) + '/' + args.model_type+ '_lr_' + str(args.learning_rate) + 'seed_' + str(args.seed) + 'decay_' + str(args.decay) +'/'
if not os.path.exists(super_folder+'argument_models/' + save_folder_name):
    os.makedirs(super_folder+'argument_models/' + save_folder_name)
    
accuracy_dictionary = {'training_loss': [], 'training':[], 'test':[]}
start_train_time = time.time()
for t in range(20):
    model.train()
    if t>0 and (t+1)%args.decay_term ==0:
        learning_rate *= args.decay
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    avg_training_loss = 0.0
    training_acc = 0.0
    for b in range(int(input_ids.shape[0]/N)):
        input_ids_batch = input_ids[N*b:N*(b+1)].to(device)
        labels_batch = label_train[N*b:N*(b+1)].to(device)
        attention_mask_batch = attention_mask[N*b:N*(b+1)].to(device)
        optimizer.zero_grad()
        #forward pass
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch.view(1,-1))
        loss = outputs.loss
        num_acc = accurate_total(y_pred=outputs.logits, y_batch=labels_batch.view(-1))
        #if t ==0:
        #    print("loss at step ", t, " : ", loss.item())
        loss.backward()
        optimizer.step()
        #
        avg_training_loss += loss
        training_acc += num_acc 
    avg_training_loss *= 1/int(input_ids.shape[0]/N)
    #Print & Evaluate
    if args.verbose:
        print("loss at step ", t, " : ", loss.item())
        print("training accuracy: ", training_acc/input_ids.shape[0])
    #evaluate
    with torch.no_grad():
        model.eval()
        outputs_val_seen = model(input_ids_val_seen, attention_mask=attention_mask_val_seen)
        val_seen_acc = accuracy(y_pred=outputs_val_seen.logits, y_batch=label_val_seen.view(-1))
        if args.verbose:
            print("validation accuracy (seen): ", val_seen_acc)
        del outputs_val_seen
    
        model_name = 'epoch_' + str(t) + '.pt'
        torch.save(model.state_dict(), super_folder+'argument_models/' + save_folder_name + model_name)
        accuracy_dictionary['training_loss'].append(loss.item())
        accuracy_dictionary['training'].append(training_acc/input_ids.shape[0])
        accuracy_dictionary['test'].append(val_seen_acc)
    
#Get the highest accuracy and delete the rest 
highest_test = np.argwhere(accuracy_dictionary['test'] == np.amax(accuracy_dictionary['test']))
highest_test = highest_test.flatten().tolist()

training_acc_highest_h = np.argmax([accuracy_dictionary['training'][h].detach().cpu().numpy() for h in highest_test])
best_t = highest_test[training_acc_highest_h]
print("The best model is 'epoch_", str(best_t)+ '.pt')

#Delete every model except for best_t 
file_path = super_folder+'argument_models/' + save_folder_name + "epoch_" + str(best_t) + ".pt"
if os.path.isfile(file_path):
    for CleanUp in glob.glob(super_folder+'argument_models/' + save_folder_name + '*.pt'):
        if not CleanUp.endswith(file_path):    
            os.remove(CleanUp)

#Save training/ test accuracy dictionary in a txt file
f = open(super_folder+ 'argument_models/' + save_folder_name +  "training_log.txt", "w")
for t in range(100):
    if t == best_t:
        f.write("===========================================\n")
    f.write("Epoch " + str(t) + "\n")
    f.write("training loss: " + str(accuracy_dictionary['training_loss'][t]) + "\n")
    f.write("training accuracy: " + str(accuracy_dictionary['training'][t]) + "\n")
    f.write("validation (seen) accuracy: " + str(accuracy_dictionary['test'][t]) + "\n")
    if t == best_t:
        f.write("===========================================")
f.close()

#Print training/ test accuracy for t
print("Saved and finished")

