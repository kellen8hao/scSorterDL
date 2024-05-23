#!/usr/bin/env python
# coding: utf-8

# # scSorterDL LDA implementation


import torch
from torch import nn

## LDA model

class LDA(nn.Module):
    def __init__(self, gene_indices, cell_indices, total_cell_types=10, priors=None,  reg_param=0.1, device='cuda',local_reg=False,shrinkage=True):
        super(LDA, self).__init__()
        self.priors = torch.as_tensor(priors,device=device) if priors is not None else None
        self.reg_param = reg_param
        self.genes=gene_indices #torch.as_tensor(gene_indices,device=device)
        self.cells=cell_indices #torch.as_tensor(cell_indices,device=device)
        self.cell_types=None
        self.device=device
        self.softmax=nn.Softmax(dim=1)
        self.total_cell_types=total_cell_types
        self.shrinkage=shrinkage
        self.local_reg=local_reg
        #print(f'LDA init - local_reg: {local_reg}, {self.local_reg}')
        #print(f'LDA init - shrinkage: {shrinkage}, {self.shrinkage}')
    @torch.no_grad()
    def fit(self, X, y, store_covariances=False, tol=1.0e-4):
        self.cell_types,self.cell_type_counts=torch.unique(y, sorted=True,return_counts=True)
        self.n_cells,self.n_genes=X.shape
        self.n_cell_types=len(self.cell_types)
        self.means=torch.zeros(self.n_cell_types,self.n_genes,device=X.device)
        covs=torch.zeros(self.n_genes,self.n_genes,device=X.device)
        self.covariance=torch.zeros(self.n_genes,self.n_genes,device=X.device)
        if self.priors is None:
            self.priors=self.cell_type_counts/self.n_cells
        for c,celltype in enumerate(self.cell_types):
            Xg = X[y == celltype, :]
            self.means[c,:]=torch.mean(Xg,dim=0,keepdim=True)
            Xg_c=Xg-self.means[c,:]
            covs=1./(self.cell_type_counts[c])*torch.matmul(Xg_c.T,Xg_c)
            if self.local_reg:
                if self.shrinkage:
                    norm_trace=torch.trace(covs)/self.n_genes
                    covs = (1-self.reg_param)*covs+self.reg_param*norm_trace*torch.eye(self.n_genes,device=X.device)
                else:
                    covs += self.reg_param*torch.eye(self.n_genes,device=X.device)
            self.covariance += self.priors[c]*covs
        self.means=self.means
        self.covariance=self.covariance/torch.sum(self.priors)
        #print(f'LDA fit - local_reg: {self.local_reg}')
        #print(f'LDA fit - shrinkage: {self.shrinkage}')
        #print(f'Before Cond Num: {torch.linalg.cond(self.covariance)}, \
                #cov shape: {self.covariance.shape}, X shape: {X.shape}')
        if not self.local_reg:
            #print(f'LDA: Global Regularization ... ')
            if self.shrinkage:
            #    ##The regularized (shrunk) covariance is given by::
            #     ###(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
            #     ###where `mu = trace(cov) / n_features`.

                norm_trace=torch.trace(self.covariance)/self.n_genes
                self.covariance = (1-self.reg_param)*self.covariance+self.reg_param*norm_trace*torch.eye(self.n_genes,device=X.device)
            else:
                self.covariance += self.reg_param*torch.eye(self.n_genes,device=X.device)
        #print(f'After Cond Num: {torch.linalg.cond(self.covariance)}')
        #The descriminant function
        # (Samples,Classes)
        self.coef = (torch.linalg.lstsq(self.covariance, self.means.T)[0].T)
        self.intercept = (-0.5 * torch.diag(torch.matmul(self.means, self.coef.T))
                           + torch.log(self.priors))

        return self
    
    @torch.no_grad()
    def forward(self, X):
        # assumes the fit is already done:
        # Produces (samples,classes) - One then can choose the max amongs the columns
        # for each sample.
        out=torch.zeros(X.shape[0],self.total_cell_types,device=X.device)
        out[:,self.cell_types]= (torch.matmul(X, self.coef.T) + self.intercept)
        #print(f'Full out: {out}')
        #print(f'Restricted out: {out[:,self.cell_types]}')
        return out
    def pred(self,X):
        return torch.argmax(self(X), axis=1)
    def pred_prob(self,X):
        return self.softmax(self(X))

# QDA model
class QDA(nn.Module):
    def __init__(self, gene_indices, cell_indices, total_cell_types=10, priors=None,id=0,  reg_param=0.0, device='cuda'):
        super(QDA, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.priors = torch.as_tensor(priors,device=device) if priors is not None else None
        self.reg_param = reg_param
        self.genes=torch.as_tensor(gene_indices,device=device)
        self.cells=torch.as_tensor(cell_indices,device=device)
        self.cell_types=None
        self.device=device
        self.softmax=nn.Softmax(dim=1)
        self.id=id
        self.total_cell_types=total_cell_types
    def fit(self, X, y, store_covariances=False, tol=0.0):

        self.cell_types,self.cell_type_counts=torch.unique(y, sorted=True,return_counts=True)
        self.n_cells,self.n_genes=X.shape
        self.n_cell_types=len(self.cell_types)
        self.means=torch.zeros(self.n_cell_types,self.n_genes,device=X.device)
        self.covs=torch.zeros(self.n_cell_types,self.n_genes,self.n_genes,device=X.device)
        self.scalings=torch.zeros(self.n_cell_types,self.n_genes,device=X.device)
        self.rotations=torch.zeros(self.n_cell_types,self.n_genes,self.n_genes,device=X.device)
        #self.covariance=torch.zeros(self.n_genes,self.n_genes,device=X.device)
        if self.priors is None:
            self.priors=self.cell_type_counts.double()/self.n_cells
        for c,celltype in enumerate(self.cell_types):
            Xg = X[y == celltype, :]
            self.means[c,:]=torch.mean(Xg,dim=0,keepdim=True)
            Xg_c=Xg-self.means[c,:]
            U, S, Vt = torch.linalg.svd(Xg_c, full_matrices=False)
            self.rank = torch.sum(S > tol).int()
            if self.rank < self.n_genes:
                warnings.warn("Variables collinear...")
                #warnings.warn(f"Variables are collinear for QDA={self.id}, and cell_type={celltype} ")

                #print(self.genes)
                #if self.reg_param == 0.:
                #    warnings.warn("Setting the regularization parameter to 0.001 ")
                #    self.reg_param=0.001
                #S=S[:self.rank]
                #Vt=Vt[:self.rank,:self.rank]
            S2 = (S ** 2) / (self.cell_type_counts[c]-1.)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            self.scalings[c,:]=S2
            self.rotations[c,:,:]=Vt.T
            self.covs[c,:,:]= torch.matmul(S2 * Vt.T, Vt) #1./(self.cell_type_counts[c]-1.)*torch.matmul(Xg_c.T,Xg_c)
            #self.covariance += self.priors[c]*self.covs[c,:,:]
        #self.covariance=self.covariance/torch.sum(self.priors)
        #self.covariance += self.reg_param*torch.eye(self.n_genes,device=X.device)
        #The descriminant function requires x from the left and right - it is done
        #in forward - no coefs nor intercept here.
        return self
    def forward(self, X):
        # assumes the fit is already done:
        # Produces (samples,classes) - One then can choose the max amongs the columns
        # for each sample.
        out=torch.zeros(X.shape[0],self.total_cell_types,device=X.device).double()
        u=torch.zeros(self.total_cell_types,device=X.device).double()
        for c in self.cell_types:
            V = self.rotations[c,:,:]
            S = self.scalings[c,:]
            Sidx=S>0
            Sg=S[Sidx]
            u[c]=torch.sum(torch.log(Sg))
            Xc = X - self.means[c,:]
            #print(torch.cat((Sg ** (-0.5),S[~Sidx])))
            X2 = torch.matmul(Xc, V * (torch.cat((Sg ** (-0.5),S[~Sidx]))))
            out[:,c]=torch.sum(X2 ** 2, 1)
        res=(-0.5 * (out + u) + torch.log(self.priors)) # (samples,cell types)
        return res/torch.sum(res,1)[:,None]

    def predict(self,X):
        return torch.argmax(self.forward(X), dim=1)
    def predict_proba(self,X):
        #values=self.forward(X)
        #likelihood = torch.exp(values - values.amax(dim=1)[:, None])
        # compute posterior probabilities
        #return likelihood / likelihood.sum(dim=1)[:, None]
        return self.softmax(self.forward(X))
