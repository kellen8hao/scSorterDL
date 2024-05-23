#!/usr/bin/env python
# coding: utf-8

# scSorterDL samplers 
## Use to sample genes and cells according to different schemes

import torch

def sample_XYcells(X, y, num_of_samples):
    """
    Sample the observations (cells in our case) - this is for bagging.
    """
    sampled_indices = torch.randint(high=X.shape[0], size=(num_of_samples,))
    sampled_X = X[sampled_indices,:]
    sampled_y = y[samples_indices]

    return sampled_X, sampled_y, sampled_indices

def sample_cells(total_cells,num_of_samples,method='uniform',prob=None,device='cuda'):
    
    """
    Sample the observations (cells in our case) - this is for bagging.
    """
    if method=='balanced':
        return sample_cells_prob(total_cells,num_of_samples,prob,device=device)
    return sample_cells_wr(total_cells,num_of_samples,device=device)

def sample_cells_wr(total_cells,num_of_samples,device='cuda'):
    sampled_indices = torch.randperm(total_cells)[:num_of_samples]
    return (torch.sort(sampled_indices))[0]

def sample_cells_prob(total_cells,num_of_samples,prob,device='cuda'):
    sampled_cells = prob.multinomial(num_samples=num_of_samples, replacement=False)
    return (torch.sort(sampled_cells))[0]

def out_of_sample_cell_indices(sampled_indices,num_of_samples):
    sample_counts = torch.bincount(sampled_indices, minlength=num_of_samples)
    unsampled_indices_mask = sample_counts == 0
    indices = torch.arange(num_of_samples)
    return indices [unsampled_indices_mask]

def sample_genes(total_genes,num_of_genes,method='uniform',cell_types=None,prob=None,device='cuda'):
    """
    Sample features (genes in our case) - this like random forests - return samples gene_indices.
    """
    if method=='uniform':
        return sample_genes_wr(total_genes,num_of_genes,device=device)
    if method=='gene_info':
        return sample_genes_prob_info(total_genes,num_of_genes,cell_types,prob,device='cuda')
    return sample_genes_prob(total_genes,num_of_genes,prob,device=device)

def sample_genes_wr(total_genes,num_of_genes,device='cuda'):
    sampled_genes = torch.randperm(total_genes)[:num_of_genes]
    return (torch.sort(sampled_genes))[0]

def sample_genes_prob(total_genes,num_of_genes,prob,device='cuda'):
    sampled_genes = prob.multinomial(num_samples=num_of_genes, replacement=False)
    #sampled_genes = torch.randperm(total_genes)[:num_of_genes]
    return (torch.sort(sampled_genes))[0]

def sample_genes_prob_info(total_genes,num_of_genes,cell_types,gene_info,device='cuda'):
    num_of_genes_per_celltype=num_of_genes//cell_types.shape[0]
    sampled_genes=torch.zeros(num_of_genes_per_celltype*cell_types.shape[0],dtype=torch.int64,device=device)
    st_idx=0
    for ct in cell_types:
        sampled_genes[st_idx:st_idx+num_of_genes_per_celltype] = gene_info[:,ct].multinomial(num_samples=num_of_genes_per_celltype, replacement=False)
        st_idx += num_of_genes_per_celltype

    #sampled_genes = torch.randperm(total_genes)[:num_of_genes]
    return (torch.sort(sampled_genes))[0]

