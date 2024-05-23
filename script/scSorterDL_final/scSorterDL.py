#!/usr/bin/env python
# coding: utf-8

# # Swarm Tiny LDAs for TM Full Cell Type Classification
#
# Below, is our python code that relies on pytorch to implement our Swarm Tiny LDAs.

# In[1]:


import torch
from torch import nn
from samplers import *
from da import LDA
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit,GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset,TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
#import rpy2.robjects as robjects
#from rpy2.robjects import pandas2ri
import pandas as pd
import pickle
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import bz2
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay


class DeepSwarmLDA(nn.Module):
    def __init__(self,lda_num_of_genes=None,lda_num_of_cells=1000,num_of_ldas=1000,num_of_celltypes=10):
        super(DeepSwarmLDA, self).__init__()
        self.num_of_samples = lda_num_of_cells
        self.num_of_genes = lda_num_of_genes
        self.num_of_ldas = num_of_ldas
        self.num_of_celltypes = num_of_celltypes
        self.lda_aggregation0=nn.Linear(num_of_ldas,100,dtype=torch.float32)
        self.lda_aggregation0.weight.data.fill_(1./num_of_ldas)
        #self.dp0=nn.Dropout(0.1)
        #self.bn0=nn.BatchNorm1d(num_of_celltypes,dtype=torch.float32)
        self.lda_activation0=nn.GELU()

        #self.lda_aggregation1=nn.Linear(100,100,dtype=torch.float32)
        #self.lda_aggregation1.weight.data.fill_(0.01)
        #self.bn1=nn.BatchNorm1d(num_of_celltypes,dtype=torch.float32)
        #self.lda_activation1=nn.GELU()

        self.lda_aggregation1=nn.Linear(100,10,dtype=torch.float32)
        self.lda_aggregation1.weight.data.fill_(0.01)
        #self.bn2=nn.BatchNorm1d(num_of_celltypes,dtype=torch.float32)
        self.lda_activation1=nn.GELU()

        self.lda_aggregation2=nn.Linear(10,1,dtype=torch.float32)
        self.lda_aggregation2.weight.data.fill_(0.1)
        #self.softmax=nn.Softmax(dim=1) - no softmax as it is include in cross-entropy
        self.loss_fn = nn.CrossEntropyLoss()
    
    @torch.no_grad()
    def fit(self,X,y,device='cuda'):
        self.ldas=[]
        for i in torch.arange(self.num_of_ldas):
            gene_indices=sample_genes(X.shape[1],self.num_of_genes)
            cell_indices=sample_cells(X.shape[0],self.num_of_samples)
            lda=LDA(gene_indices,cell_indices,total_cell_types=self.num_of_celltypes).to(device)
            lda=lda.fit(X[cell_indices[:,None],gene_indices],y[cell_indices])
            self.ldas.append(lda)
        return self
    def forward(self,X):
        ldas_out=torch.zeros(X.shape[0],self.num_of_celltypes,self.num_of_ldas,device=X.device,dtype=torch.float32)
        for i,lda in enumerate(self.ldas):
            ldas_out[:,:,i]=lda.forward(X[:,lda.genes])

        out=self.lda_aggregation0(ldas_out)
        #out=self.bn0(out)
        out=self.lda_activation0(out)

        #out=self.lda_aggregation1(out)
        #out=self.bn1(out)
        #out=self.lda_activation1(out)

        out=self.lda_aggregation1(out)
        #out=self.bn2(out)
        out=self.lda_activation1(out)

        out=self.lda_aggregation2(out)
        return out #- no softmax as it is include in cross-entropy

    def train_nn(self, X,y):
        out = self(X).reshape(X.shape[0],-1)        # Generate predictions
        #print(out)
        #print(y)
        loss = self.loss_fn(out, y) # Calculate loss
        return loss,out
    def oob_error(self,X,y):
        pass
    def validation_step(self, X,y):
        out = self(X).reshape(X.shape[0],-1)
        loss = self.loss_fn(out, y)
        acc = self.accuracy(out, y)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    def accuracy(self, outputs, y):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == y).item() / len(preds))
    def evaluate(self,val_loader,device='cuda'):
        outputs = [self.validation_step(X.to(device),y.to(device)) for X,y in val_loader]
        return self.validation_epoch_end(outputs)
    def pred(self,X):
        return torch.argmax(torch.softmax(self.forward(X),axis=1), axis=1)
    def pred_prob(self,X):
        return torch.softmax(self.forward(X),axis=1)

class SwarmLDA(nn.Module):
    def __init__(self,lda_num_of_genes=None,lda_num_of_cells=1000,num_of_ldas=1000,
                  num_of_celltypes=10,common_cell_sampler=False):
        super(SwarmLDA, self).__init__()
        self.common_cell_sampler=common_cell_sampler
        self.num_of_samples = lda_num_of_cells
        self.num_of_genes = lda_num_of_genes
        self.num_of_ldas = num_of_ldas
        self.num_of_celltypes = num_of_celltypes
        self.lda_aggregation=nn.Linear(num_of_ldas,1,dtype=torch.float32)
        self.lda_aggregation.weight.data.fill_(1./num_of_ldas)
        #self.softmax=nn.Softmax(dim=1) - no softmax as it is include in cross-entropy
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def fit(self,X,y,stratified_cell_indices=torch.empty(0,dtype=torch.int),ldareg=0.1,device='cuda'):
        self.ldas=[]
        if self.common_cell_sampler:
            if stratified_cell_indices.nelement() == 0:
                cell_indices=sample_cells(X.shape[0],self.num_of_samples)
            else:
                cell_indices=stratified_cell_indices[0,:]
        for i in torch.arange(self.num_of_ldas):
            gene_indices=sample_genes(X.shape[1],self.num_of_genes)
            if not self.common_cell_sampler and stratified_cell_indices.nelement() != 0:
                cell_indices=stratified_cell_indices[i,:]
            elif not self.common_cell_sampler:
                cell_indices=sample_cells(X.shape[0],self.num_of_samples)
            lda=LDA(gene_indices,cell_indices,total_cell_types=self.num_of_celltypes,reg_param=ldareg).to(device)
            lda=lda.fit(X[cell_indices[:,None],gene_indices],y[cell_indices])
            self.ldas.append(lda)
            #print('X shape, fit: ',X[cell_indices[:,None],gene_indices].shape)
            #print('Y shape, fit: ',y[cell_indices].shape)
            #print('c shape, fit: ',cell_indices.shape)
        #print("Memory: ",torch.cuda.memory_allocated(device))
        del stratified_cell_indices
        return self
    def forward(self,X):
        ldas_out=torch.zeros(X.shape[0],self.num_of_celltypes,self.num_of_ldas,device=X.device,dtype=torch.float32)
        for i,lda in enumerate(self.ldas):
            ldas_out[:,:,i]=lda.forward(X[:,lda.genes])

        out=self.lda_aggregation(ldas_out)
        return out #- no softmax as it is included in cross-entropy

    def train_nn(self, X,y):
        out = self(X).reshape(X.shape[0],-1)        # Generate predictions
        #print(out)
        #print(y)
        loss = self.loss_fn(out, y) # Calculate loss
        return loss,out
    def oob_error(self,X,y):
        pass
    def validation_step(self, X,y):
        out = self(X).reshape(X.shape[0],-1)
        loss = self.loss_fn(out, y)
        acc = self.accuracy(out, y)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    def accuracy(self, outputs, y):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == y).item() / len(preds))
    def evaluate(self,val_loader,device='cuda'):
        outputs = [self.validation_step(X.to(device),y.to(device)) for X,y in val_loader]
        return self.validation_epoch_end(outputs)
    def pred(self,X):
        return torch.argmax(torch.softmax(self.forward(X),axis=1), axis=1)
    def pred_prob(self,X):
        return torch.softmax(self.forward(X),axis=1)

class WebSwarmLDA(nn.Module):
    def __init__(self,lda_num_of_genes=None,lda_num_of_cells=1000,num_of_ldas=1000,
                  num_of_celltypes=10,web_layer_sizes=(10,20,10),common_cell_sampler=False):
        super(WebSwarmLDA, self).__init__()
        self.common_cell_sampler=common_cell_sampler
        self.num_of_samples = lda_num_of_cells
        self.num_of_genes = lda_num_of_genes
        self.num_of_ldas = num_of_ldas
        self.num_of_celltypes = num_of_celltypes
        self.lda_aggregation=nn.Linear(num_of_ldas,1,dtype=torch.float32)
        #self.lda_aggregation.weight.data.fill_(1./num_of_ldas)
        #self.lda_aggregation.weight.data.fill_(1.)
        #self.softmax=nn.Softmax(dim=1) - no softmax as it is include in cross-entropy
        # add nonlinearity DNN to each LDA output
        self.lda_out=None
        self.weblda=None
        self.ldas_cell_type_counts=torch.zeros(num_of_celltypes)
        if web_layer_sizes and len(web_layer_sizes)>0:
            self.weblda=nn.ModuleList()
            for i in range(num_of_ldas):
                ldaextension=nn.ModuleList()
                # The first layer
                #ldaextension.append(nn.ReLU())
                ldaextension.append(nn.BatchNorm1d(num_of_celltypes))
                ldaextension.append(nn.Linear(num_of_celltypes,web_layer_sizes[0]))
                ldaextension.append(nn.BatchNorm1d(web_layer_sizes[0]))
                # The activation layer
                ldaextension.append(nn.ReLU())
                for i in range(len(web_layer_sizes)-1):
                    ldaextension.append(nn.Linear(web_layer_sizes[i], web_layer_sizes[i+1]))
                    ldaextension.append(nn.BatchNorm1d(web_layer_sizes[i+1]))
                    ldaextension.append(nn.ReLU())
                ldaextension.append(nn.Linear(web_layer_sizes[-1],num_of_celltypes))
                self.weblda.append(nn.Sequential(*ldaextension))

        #
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    @torch.no_grad()
    def fit(self,X,y,stratified_cell_indices=torch.empty(0,dtype=torch.int64),
            cell_group_split=None,ldareg=0.1,device='cuda'):
        self.ldas=[]
        gene_idcs_sampled=torch.empty(0,dtype=torch.int64)
        if self.common_cell_sampler:
            if stratified_cell_indices.nelement() == 0:
                cell_indices=sample_cells_wr(X.shape[0],self.num_of_samples)
            else:
                cell_indices=stratified_cell_indices[0,:]
        if cell_group_split:
            for i in torch.arange(0,self.num_of_ldas,2):
                gene_indices=sample_genes_wr(X.shape[1],self.num_of_genes)
                gene_idcs_sampled=torch.concat((gene_idcs_sampled,gene_indices)).unique()
                cell_indices_1,cell_indices_2=next(cell_group_split)
                cell_indices_1=torch.from_numpy(cell_indices_1)
                lda=LDA(gene_indices,cell_indices_1,total_cell_types=self.num_of_celltypes,reg_param=ldareg).to(device)
                y_proj=y[cell_indices_1]
                y_cell_types=y_proj.unique()
                self.ldas_cell_type_counts[y_cell_types]= self.ldas_cell_type_counts[y_cell_types]+1
                lda=lda.fit(X[cell_indices_1[:,None],gene_indices],y[cell_indices_1])
                self.ldas.append(lda)
                gene_indices=sample_genes_wr(X.shape[1],self.num_of_genes)
                cell_indices_2=torch.from_numpy(cell_indices_2)
                y_proj=y[cell_indices_2]
                y_cell_types=y_proj.unique()
                self.ldas_cell_type_counts[y_cell_types]= self.ldas_cell_type_counts[y_cell_types]+1

                lda=LDA(gene_indices,cell_indices_2,total_cell_types=self.num_of_celltypes,reg_param=ldareg).to(device)
                lda=lda.fit(X[cell_indices_2[:,None],gene_indices],y[cell_indices_2])
                self.ldas.append(lda)
            del cell_group_split

        else:
            for i in torch.arange(self.num_of_ldas):
                gene_indices=sample_genes_wr(X.shape[1],self.num_of_genes)
                gene_idcs_sampled=torch.concat((gene_idcs_sampled,gene_indices)).unique()
                if not self.common_cell_sampler and stratified_cell_indices.nelement() != 0:
                    cell_indices=stratified_cell_indices[i,:]
                elif not self.common_cell_sampler:
                    cell_indices=sample_cells_wr(X.shape[0],self.num_of_samples)

                y_proj=y[cell_indices]
                y_cell_types=y_proj.unique()
                self.ldas_cell_type_counts[y_cell_types]= self.ldas_cell_type_counts[y_cell_types]+1

                lda=LDA(gene_indices,cell_indices,total_cell_types=self.num_of_celltypes,reg_param=ldareg).to(device)
                lda=lda.fit(X[cell_indices[:,None],gene_indices],y[cell_indices])
                self.ldas.append(lda)
                #print('X shape, fit: ',X[cell_indices[:,None],gene_indices].shape)
                #print('Y shape, fit: ',y[cell_indices].shape)
                #print('c shape, fit: ',cell_indices.shape)
            #print("Memory: ",torch.cuda.memory_allocated(device))
        del stratified_cell_indices
        print(f'Are all the genes sampled: {len(gene_idcs_sampled)==X.shape[1]}')
        with torch.no_grad():
            self.lda_aggregation.weight.fill_(1./len(self.ldas)) #copy_(1./self.ldas_cell_type_counts)
            self.lda_aggregation.bias.fill_(0.)
        return self
    def forward(self,X):
        ldas_out=torch.zeros(X.shape[0],self.num_of_celltypes,self.num_of_ldas,device=X.device,dtype=torch.float32)
        if self.weblda:
            self.lda_out=torch.zeros(self.num_of_ldas,X.shape[0],self.num_of_celltypes,device=X.device,dtype=torch.float32)
            for i,lda in enumerate(self.ldas):
                ldaextension=self.weblda[i]
                ldas_out[:,:,i]=ldaextension(lda.forward(X[:,lda.genes]))
                self.lda_out[i,:,:]=ldas_out[:,:,i]
        else:
            for i,lda in enumerate(self.ldas):
                ldas_out[:,:,i]=lda.forward(X[:,lda.genes])
        out=self.lda_aggregation(ldas_out)
        return out #- no softmax as it is included in cross-entropy

    def train_nn(self, X,y):
        out = self(X).reshape(X.shape[0],-1)        # Generate predictions
        #print(out)
        #print(y)
        loss = self.loss_fn(out, y) # Calculate loss
        if self.lda_out is not None:
            for i in range(self.num_of_ldas):
                loss += self.loss_fn(self.lda_out[i,:,:],y)
        return loss,out
    def oob_error(self,X,y):
        pass
    def validation_step(self, X,y):
        out = self(X).reshape(X.shape[0],-1)
        loss = self.loss_fn(out, y)
        _, preds = torch.max(out, dim=1)
        acc=torch.sum(preds == y)
        #acc = self.accuracy(out, y)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs,total_size=None):
        if total_size:
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).sum()/total_size   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).sum()/total_size      # Combine accuracies

        else:
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
    def train_epoch_end(self, epoch, result):
        print("Epoch [{}], trn_loss: {:.4f}, trn_acc: {:.4f}".format(epoch, result['tr_loss'], result['tr_acc']))
    
    def accuracy(self, outputs, y):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == y).item() / len(preds))

    @torch.no_grad()
    def evaluate(self,val_loader,device='cuda'):
        total_size=0
        outputs=[]
        for X,y in val_loader:
            outputs.append(self.validation_step(X.to(device),y.to(device)))
            total_size += len(y)
        #outputs = [self.validation_step(X.to(device),y.to(device)) for X,y in val_loader]
        return self.validation_epoch_end(outputs,total_size)
    def pred(self,X):
        return torch.argmax(torch.softmax(self.forward(X),axis=1), axis=1)
    def pred_prob(self,X):
        return torch.softmax(self.forward(X),axis=1)

class LocalSwarmLDA(nn.Module):
    def __init__(self,lda_num_of_genes=None,lda_num_of_cells=1000,num_of_ldas=1000,
                  num_of_celltypes=10,is_web=False,web_layer_sizes=(10,20,10),is_local=False,
                  common_cell_sampler=False):
        super(LocalSwarmLDA, self).__init__()
        self.common_cell_sampler=common_cell_sampler
        self.num_of_samples = lda_num_of_cells
        self.num_of_genes = lda_num_of_genes
        self.num_of_ldas = num_of_ldas
        self.num_of_celltypes = num_of_celltypes
        self.is_local=is_local
        if is_local:
            self.lda_aggregation=nn.ModuleList()
            for i in range(num_of_celltypes):
                self.lda_aggregation.append(nn.Linear(num_of_ldas,1,dtype=torch.float32))
        else:
            self.lda_aggregation=nn.Linear(num_of_ldas,1,dtype=torch.float32)
        #self.lda_aggregation.weight.data.fill_(1./num_of_ldas)
        #self.lda_aggregation.weight.data.fill_(1.)
        #self.softmax=nn.Softmax(dim=1) - no softmax as it is include in cross-entropy
        # add nonlinearity DNN to each LDA output
        self.lda_out=None
        self.weblda=None
        self.ldas_cell_type_counts=torch.zeros(num_of_celltypes)
        if is_web and len(web_layer_sizes)>0:
            self.weblda=nn.ModuleList()
            for i in range(num_of_ldas):
                ldaextension=nn.ModuleList()
                # The first layer
                #ldaextension.append(nn.ReLU())
                ldaextension.append(nn.BatchNorm1d(num_of_celltypes))
                ldaextension.append(nn.Linear(num_of_celltypes,web_layer_sizes[0]))
                ldaextension.append(nn.BatchNorm1d(web_layer_sizes[0]))
                # The activation layer
                ldaextension.append(nn.ReLU())
                for i in range(len(web_layer_sizes)-1):
                    ldaextension.append(nn.Linear(web_layer_sizes[i], web_layer_sizes[i+1]))
                    ldaextension.append(nn.BatchNorm1d(web_layer_sizes[i+1]))
                    ldaextension.append(nn.ReLU())
                ldaextension.append(nn.Linear(web_layer_sizes[-1],num_of_celltypes))
                self.weblda.append(nn.Sequential(*ldaextension))
        elif is_web and len(web_layer_sizes)==0: # Very Shallow architecture
            self.weblda=nn.ModuleList()
            for i in range(num_of_ldas):
                ldaextension=nn.ModuleList()
                #ldaextension.append(nn.Sigmoid())
                ldaextension.append(nn.BatchNorm1d(num_of_celltypes))
                ldaextension.append(nn.Linear(num_of_celltypes,num_of_celltypes))
                self.weblda.append(nn.Sequential(*ldaextension))

        #
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    @torch.no_grad()
    def fit(self,X,y,stratified_cell_indices=torch.empty(0,dtype=torch.int64),
            cell_group_split=None,local_reg=False,shrinkage=True,ldareg=0.1,
            gene_info=None, device='cuda',gene_sampling='uniform',cell_sampling='uniform'):
        self.ldas=[]
        gene_idcs_sampled=torch.empty(0,dtype=torch.int64,device=device)
        _,cell_type_counts=y.unique(return_counts=True)
        cell_prob=None
        gene_prob=None
        if cell_sampling=='balanced':
            cell_prob=1./torch.log(cell_type_counts[y]+1) # take 1./(cell_type_counts[y]+1) is you want more sampling of rare cells.
        if gene_sampling=='dirichlet':
            gene_dirAlpha,_ = torch.median(gene_info,dim=1) #
            dirDistr=torch.distributions.dirichlet.Dirichlet(gene_dirAlpha)
            gene_prob=dirDistr.sample()
        elif gene_sampling=='gene_info':
            gene_prob=gene_info
        if self.common_cell_sampler:
            if stratified_cell_indices.nelement() == 0:
                cell_indices=sample_cells_wr(X.shape[0],self.num_of_samples)
            else:
                cell_indices=stratified_cell_indices[0,:]
        if cell_group_split:
            for i in torch.arange(0,self.num_of_ldas,2):
                gene_indices=sample_genes(X.shape[1],self.num_of_genes,method=gene_sampling,prob=gene_prob)
                gene_idcs_sampled=torch.concat((gene_idcs_sampled,gene_indices)).unique()
                cell_indices_1,cell_indices_2=next(cell_group_split)
                cell_indices_1=torch.from_numpy(cell_indices_1)
                lda=LDA(gene_indices,cell_indices_1,total_cell_types=self.num_of_celltypes,reg_param=ldareg,local_reg=local_reg,shrinkage=shrinkage).to(device)
                y_proj=y[cell_indices_1]
                y_cell_types=y_proj.unique()
                self.ldas_cell_type_counts[y_cell_types]= self.ldas_cell_type_counts[y_cell_types]+1
                lda=lda.fit(X[cell_indices_1[:,None],gene_indices],y[cell_indices_1])
                self.ldas.append(lda)
                #gene_indices=sample_genes_wr(X.shape[1],self.num_of_genes)
                cell_indices_2=torch.from_numpy(cell_indices_2)
                y_proj=y[cell_indices_2]
                y_cell_types=y_proj.unique()
                self.ldas_cell_type_counts[y_cell_types]= self.ldas_cell_type_counts[y_cell_types]+1

                lda=LDA(gene_indices,cell_indices_2,total_cell_types=self.num_of_celltypes,reg_param=ldareg,local_reg=local_reg,shrinkage=shrinkage).to(device)
                lda=lda.fit(X[cell_indices_2[:,None],gene_indices],y[cell_indices_2])
                self.ldas.append(lda)
            del cell_group_split

        else:
            for i in torch.arange(self.num_of_ldas):
                # Sample the cells first
                if not self.common_cell_sampler and stratified_cell_indices.nelement() != 0:
                    cell_indices=stratified_cell_indices[i,:]
                elif not self.common_cell_sampler:
                    #cell_indices=sample_cells_wr(X.shape[0],self.num_of_samples)
                    cell_indices=sample_cells(X.shape[0],self.num_of_samples,method=cell_sampling,prob=cell_prob)
                    #print(f'cell_indices:\n {cell_indices.cpu()}')
                y_proj=y[cell_indices]
                y_cell_types=y_proj.unique()
                self.ldas_cell_type_counts[y_cell_types]= self.ldas_cell_type_counts[y_cell_types]+1
                # Sample the genes
                #gene_indices=sample_genes_wr(X.shape[1],self.num_of_genes)
                if gene_sampling=='dirichlet': # resample the sampling distribution if dirichlet is used.
                    if i%10==0:
                        gene_prob=dirDistr.sample()
                gene_indices=sample_genes(X.shape[1], self.num_of_genes,cell_types=y_cell_types,method=gene_sampling,prob=gene_prob)

                lda=LDA(gene_indices,cell_indices,total_cell_types=self.num_of_celltypes,reg_param=ldareg).to(device)
                lda=lda.fit(X[cell_indices[:,None],gene_indices],y[cell_indices])
                self.ldas.append(lda)
                #print('X shape, fit: ',X[cell_indices[:,None],gene_indices].shape)
                #print('Y shape, fit: ',y[cell_indices].shape)
                #print('c shape, fit: ',cell_indices.shape)
            #print("Memory: ",torch.cuda.memory_allocated(device))
        del stratified_cell_indices
        print(f'Are all the genes sampled: {len(gene_idcs_sampled)==X.shape[1]}')
        with torch.no_grad():
            if self.is_local:
                for i in range(self.num_of_celltypes):
                    self.lda_aggregation[i].weight.fill_(1./len(self.ldas)) #copy_(1./self.ldas_cell_type_counts)
                    self.lda_aggregation[i].bias.fill_(0.)
            else:
                self.lda_aggregation.weight.fill_(1./len(self.ldas)) #copy_(1./self.ldas_cell_type_counts)
                self.lda_aggregation.bias.fill_(0.)
        return self
    def forward(self,X):
        ldas_out=torch.zeros(X.shape[0],self.num_of_celltypes,self.num_of_ldas,device=X.device,dtype=torch.float32)
        if self.weblda:
            self.lda_out=torch.zeros(self.num_of_ldas,X.shape[0],self.num_of_celltypes,device=X.device,dtype=torch.float32)
            for i,lda in enumerate(self.ldas):
                ldaextension=self.weblda[i]
                ldas_out[:,:,i]=ldaextension(lda.forward(X[:,lda.genes]))
                self.lda_out[i,:,:]=ldas_out[:,:,i]
        else:
            for i,lda in enumerate(self.ldas):
                ldas_out[:,:,i]=lda.forward(X[:,lda.genes])
        if self.is_local:
            out=torch.zeros(X.shape[0],self.num_of_celltypes,device=X.device)
            for i in range(self.num_of_celltypes):
                #print('-----------ldas_out shape: ',ldas_out[:,i,:].shape)
                #print('------------local out shape: ',self.lda_aggregation[i](ldas_out[:,i,:]).shape)
                out[:,i]=self.lda_aggregation[i](ldas_out[:,i,:]).reshape(-1)
        else:
            out=self.lda_aggregation(ldas_out)
        #print('-------------Output shape for local: ',out.shape)
        return out #- no softmax as it is included in cross-entropy

    def train_nn(self, X,y):
        out = self(X).reshape(X.shape[0],-1)        # Generate predictions
        #print(out)
        #print(y)
        num_of_samples=len(y)
        loss = self.loss_fn(out, y)/num_of_samples # Calculate avg loss
        if self.lda_out is not None:
            for i in range(self.num_of_ldas):
                loss += self.loss_fn(self.lda_out[i,:,:],y)/num_of_samples
        return loss,out
    def oob_error(self,X,y):
        pass
    # This keeps track of the total so far of the loss and accuracy
    def validation_step(self, X,y):
        out = self(X).reshape(X.shape[0],-1)
        loss = self.loss_fn(out, y)
        _, preds = torch.max(out, dim=1)
        acc=torch.sum(preds == y)
        #acc = self.accuracy(out, y)
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs,total_size=None):
        if total_size:
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).sum()/total_size   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).sum()/total_size      # Combine accuracies
        else:
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, 
            result['val_loss'], result['val_acc']), flush=True)
    
    def train_epoch_end(self, epoch, result):
        print("Epoch [{}], trn_loss: {:.4f}, trn_acc: {:.4f}".format(epoch,
            result['tr_loss'], result['tr_acc']),flush=True)
    
    def accuracy(self, outputs, y):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == y).item() / len(preds))
    @torch.no_grad()
    def evaluate(self,val_loader,device='cuda'):
        total_size=0
        outputs=[]
        for X,y in val_loader:
            outputs.append(self.validation_step(X.to(device),y.to(device)))
            total_size += len(y)
        #outputs = [self.validation_step(X.to(device),y.to(device)) for X,y in val_loader]
        return self.validation_epoch_end(outputs,total_size)
    def pred(self,X):
        return torch.argmax(torch.softmax(self.forward(X),axis=1), axis=1)
    def pred_prob(self,X):
        return torch.softmax(self.forward(X),axis=1)
