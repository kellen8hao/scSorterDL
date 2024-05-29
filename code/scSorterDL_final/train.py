#!/usr/bin/env python
# coding: utf-8

# # Swarm Tiny LDAs for TM Full Cell Type Classification
#
# Below, is our python code that relies on pytorch to implement our Swarm Tiny LDAs.

# In[1]:


import torch
from da import LDA
from utils import *
from scSorterDL import *
from samplers import *
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
import time
import matplotlib.pyplot as plt
import sys
import copy
import bz2
import seaborn as sn
import os
def train_swarmlda(model, train_loader, val_loader, ofval_loader, optimizer=torch.optim.SGD, n_epochs=1, 
        log_every=5, lr=0.001,device='cuda',best_model=None, best_val_loss=1000., best_val_acc=1.0, 
        scheduler=None,T_0=200,T_mult=1,verbose=False,use_scheduler=False):
    optimizer = optimizer(model.parameters(), lr)
    if use_scheduler and not scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=T_0, 
                T_mult=T_mult, verbose=verbose)
    if not best_model:
        best_model=copy.deepcopy(model)
        model.eval()
        with torch.no_grad():
            result = model.evaluate(ofval_loader)
        best_val_loss = result['val_loss']
        best_val_acc = result['val_acc']
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    history = [] # for recording epoch-wise ifval results
    ofhistory = [] # for recording epoch-wise ofval results
    trhistory =[]  # for recording epoch-wise training loss
    for epoch in range(n_epochs):
        model.train()
        tb=0
        trloss=0.
        tracc =0.
        # Training Phase
        for batch in train_loader:
            X,y=batch
            loss,out_pred = model.train_nn(X.to(device),y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb += 1
            trloss += loss.item()
            tracc += model.accuracy(out_pred,y.to(device)).item()
        # Validation phase
        model.eval()
        #with torch.no_grad():
        #    tr_result = model.evaluate(train_loader)
        #    if epoch % log_every==0:
        #        print("---Training loss: ")
        #        model.epoch_end(epoch, tr_result)
        trloss = trloss/tb
        tracc  = tracc/tb 
        tr_result={'tr_loss':trloss,'tr_acc': tracc}
        if epoch % log_every==0:
            print("---Training loss: ")
            model.train_epoch_end(epoch,tr_result)
        trhistory.append(tr_result)
        with torch.no_grad():
            result = model.evaluate(val_loader)
            if epoch % log_every==0:
                print("---In fit validation: ")
                model.epoch_end(epoch, result)
            history.append(result)
            # out of fit validation is much better 
            # for validation and model comparison
            if ofval_loader:
                result = model.evaluate(ofval_loader)
                if epoch % log_every==0:
                    print("---Out of fit validation: ")
                    model.epoch_end(epoch, result)
            ofhistory.append(result)
        #if best_val_loss > result['val_loss'] and
        if best_val_acc < result['val_acc']:
            best_val_loss = result['val_loss']
            best_val_acc = result['val_acc']
            best_model = copy.deepcopy(model)
        if use_scheduler:
            scheduler.step()

    return history, trhistory, ofhistory, best_model, best_val_loss, best_val_acc, scheduler

def run(args):
    #set the random seed for checking the robustness of our models
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    batch_size=args.batch_size
    runID=args.runID
    trainingfile=args.train_data
    
    group_split=False # This will be done at inside the stratified portion of the code...
    stratified=False
    local_reg = not args.global_reg
    shrinkage= not args.noshrinkage
    fine_celltype_aggregation=args.fine_aggr
    loss_history=[]
    ofval_history=[]
    ofloss_history=[]
    trloss_history=[]
    device=args.device
    # When ofval_loader is None, we will consider out-of-fit validation to be the same as in-fit validation
    ofval_loader=None
    optimizers={'NAdam':torch.optim.NAdam,'AdamW':torch.optim.AdamW,'SGD':torch.optim.SGD}

    le=None
    df=None
    if device=='auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    restart=args.restart
    gene_info=None
    if restart==0:
        if args.gene_sampling=='gene_info':
            gene_info=os.path.join(os.path.dirname(trainingfile),'gene_info.pkl')
            gene_info_df=pd.read_pickle(gene_info)
        if trainingfile.endswith('.rds'): # R object stored data
            readRDS = robjects.r['readRDS']
            df = readRDS(trainingfile)
            df = pandas2ri.rpy2py_dataframe(df)
            with open(trainingfile.replace('.rds','.pkl'),'wb') as picklefile:
                pickle.dump(df,picklefile)
        else: # assume this is pickle file
            with open(trainingfile, 'rb') as pklf:
                df=pickle.load(pklf)
        print('df: ',df.head())

        le = preprocessing.LabelEncoder()
        le.fit(df.yy)
        df['yy_int'] = le.transform(df.yy)
        print(f'LabelEncoder: {list(le.classes_)}')
        if gene_info:
            print(f'Gene Info Columns: {gene_info_df.columns}')
            gene_info_df.columns=le.transform(gene_info_df.columns) # ensure the same encoding for the cell type in both
                                                                    # data set and gene info matrix.
            gene_info_df=gene_info_df.reindex(sorted(gene_info_df.columns),axis=1)
            gene_names=df.columns[~df.columns.isin(['yy','yy_int'])]
            gene_info_df=gene_info_df.loc[gene_names]      # ensure the same ordering on of the genes - genes in the index
                                                            # of gene_info should be in the same order as df columns.
            print(gene_info_df.head())
            np_gen_info_vals=np.float32(gene_info_df.values)
            np_gen_info_vals[np_gen_info_vals==np.inf]=5*max(np_gen_info_vals[np_gen_info_vals!=np.inf])
            gene_info=torch.from_numpy(np_gen_info_vals).to(device)
        #le.inverse_transform(df['yy_int'])
        allX,ally=df.loc[:,~df.columns.isin(['yy','yy_int'])].values.astype(np.float32),df.yy_int.values.astype(np.int64)
        print('allX shape: ',allX.shape)
        print(df['yy_int'].value_counts(sort=True, ascending=False))
        ##skfold=StratifiedKFold(n_splits=6, shuffle=False)
        ##for train_index, test_index in skfold.split(allX, ally):
        # Generate training for the tinyLDA- to fit each lda.
        # Generate validation for the tinyLDA to check our optimized LDA aggregation weights
        #X_fit, X_tr_val_ts, y_fit, y_tr_val_ts = train_test_split(allX, ally, test_size=0.5, stratify=ally)
       #X_tr, X_val, y_tr, y_val = train_test_split(X_tr_val_ts, y_tr_val_ts, test_size=0.10, stratify=y_tr_val_ts)
        if args.retrain:
            #X_val, X_test, y_val, y_test=torch.from_numpy(X_val),torch.from_numpy(X_ts),torch.from_numpy(y_val),torch.from_numpy(y_ts)
            X_fit = allX
            y_fit = ally
            X_tr, X_val, y_tr, y_val = train_test_split(allX, ally, test_size=0.15, stratify=ally)
            print('X_fit shape: ', X_fit.shape)
            print('X_tr shape: ', X_tr.shape)
            print('X_val shape: ',X_val.shape)
        else:
            X_fit, X_left, y_fit, y_left = train_test_split(allX, ally, test_size=0.20, stratify=ally)
            #from imblearn.over_sampling import RandomOverSampler
            #ros=RandomOverSampler()
            #XROS_left,yROS_left=ros.fit_resample(X_left,y_left)
            #X_mrtr, X_mrval, y_mrtr, y_mrval = train_test_split(XROS_left, yROS_left, test_size=0.20, stratify=yROS_left)
            splitIdx=np.int64(X_left.shape[0]*0.8)
            X_mrtr, X_mrval, y_mrtr, y_mrval = X_left[:splitIdx,:],X_left[splitIdx:,:],y_left[:splitIdx],y_left[splitIdx:]
            X_tr, X_val, y_tr, y_val = train_test_split(X_fit, y_fit, test_size=0.10, stratify=y_fit)
            #X_val, X_ts, y_val, y_ts = train_test_split(X_val_ts, y_val_ts, test_size=0.10, stratify=y_val_ts)
            X_tr=np.concatenate((X_tr,X_mrtr),axis=0)
            y_tr=np.concatenate((y_tr,y_mrtr),axis=0)
            #X_val=np.concatenate((X_val,X_mrval),axis=0)
            #y_val=np.concatenate((y_val,y_mrval),axis=0)
            print('X_fit shape: ', X_fit.shape)
            print('X_tr shape: ', X_tr.shape)
            print('X_val shape: ',X_val.shape)
            print('X_ofval shape: ',X_mrval.shape)

        # Use the same as Random Forest criterion for the number of genes but with more additional genes :-)
        
        num_of_genes_per_lda=int(np.sqrt(X_fit.shape[1]))+70 # ~ 70
        #num_of_genes_per_lda=X_fit.shape[1]//160 # ~ 70
        # Use 80% of the cells
        num_of_cells_per_lda=int(X_fit.shape[0]*args.cellsr)
        num_of_ldas=args.numldas
        print(f"Number of LDAs: {num_of_ldas}, Number of genes per LDA: {num_of_genes_per_lda}, Number of cells per LDA: {num_of_cells_per_lda}")
        stratified_cell_indices=torch.empty(0,dtype=torch.int)
        cell_group_split=None
        if args.cell_sampling=='group_split':
            group_split=True
        if args.cell_sampling=='stratified':
            stratified=True
        if group_split or stratified:
            if group_split:
                gss = GroupShuffleSplit(n_splits=num_of_ldas, test_size=0.5)
                cell_group_split=gss.split(X_fit, y_fit,groups=y_fit)
            else:
                if num_of_cells_per_lda >= X_fit.shape[0]:
                    num_of_cells_per_lda=X_fit.shape[0]
                    stratified_cell_indices=torch.arange(num_of_cells_per_lda).reshape(1,-1)
                else:
                    sss = StratifiedShuffleSplit(n_splits=num_of_ldas, train_size=num_of_cells_per_lda)
                    stratified_cell_indices=torch.from_numpy(np.array([s for s,t in sss.split(X_fit, y_fit)]))
                    del sss
            print(f'stratified_cell_indices.shape:{stratified_cell_indices.shape}')
            #stratified_cell_indices=torch.from_numpy(stratified_cell_indices)
        #Convert to torch tensor
        X_fit, y_fit=torch.from_numpy(X_fit),torch.from_numpy(y_fit)
        X_train, y_train=torch.from_numpy(X_tr),torch.from_numpy(y_tr)
        X_val, y_val=torch.from_numpy(X_val),torch.from_numpy(y_val)
        if not args.retrain:
            X_ofval, y_ofval=torch.from_numpy(X_mrval),torch.from_numpy(y_mrval)

        print(f'Training classes:\n    {torch.unique(y_train, sorted=True,return_counts=True)}')
        print(f'Validation classes:\n   {torch.unique(y_val, sorted=True,return_counts=True)}')
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
        if args.retrain:
            ofval_loader=None
        else:
            ofval_dataset = TensorDataset(X_ofval, y_ofval)
            ofval_loader = DataLoader(ofval_dataset, batch_size=batch_size,shuffle=False)

    else:
        with bz2.open('ckpt_'+runID+'.bz2','rb') as fh:
            print('Loading last model ...')
            swarm_ldas=torch.load('ckpt_'+runID+'_last.pt')
            print('Loading best model ...')
            best_model=torch.load('ckpt_'+runID+'_best.pt')
            print('Loading initial model ...')
            init_swarm_ldas=torch.load('ckpt_'+runID+'_init.pt')
            print('Loading the rest of pickled variables ...')
            [train_loader, val_loader, ofval_loader, loss_history, trloss_history, 
                    ofloss_history, res_df,trres_df, ofval_history,ofval_df,scheduler,
                    best_val_loss,best_val_acc,le] = pickle.load(fh)


    # In[22]:
    if restart==0:
        num_of_celltypes=len(torch.unique(y_fit))
        web_layer_sizes=args.web_layer_sizes
        if web_layer_sizes is None:
            web=False
        else:
            web=True
        swarm_ldas=LocalSwarmLDA(lda_num_of_genes=num_of_genes_per_lda,
                        lda_num_of_cells=num_of_cells_per_lda,
                        num_of_ldas=num_of_ldas,num_of_celltypes=num_of_celltypes,
                        common_cell_sampler=args.common_cell_sampler,
                        web_layer_sizes=web_layer_sizes,is_web=web,
                        is_local=fine_celltype_aggregation).to(device)

        swarm_ldas.fit(X_fit.to(device),y_fit.to(device),
                                   stratified_cell_indices=stratified_cell_indices.to(device),
                                   cell_group_split=cell_group_split,ldareg=args.ldareg,
                                   local_reg=local_reg,shrinkage=shrinkage, gene_info=gene_info,cell_sampling=args.cell_sampling,gene_sampling=args.gene_sampling)
        count_parameters(swarm_ldas)
        init_swarm_ldas=copy.deepcopy(swarm_ldas) 
    print('swarm_ldas.ldas[0].cells:\n',swarm_ldas.ldas[0].cells)
    print('swarm_ldas.ldas[0].genes:\n',swarm_ldas.ldas[0].genes)
    print('swarm_ldas.ldas[1].cells:\n',swarm_ldas.ldas[1].cells)
    print('swarm_ldas.ldas[1].genes:\n',swarm_ldas.ldas[1].genes)

    torch.cuda.empty_cache()
    if restart==0:
       with torch.no_grad():
            lda=swarm_ldas.ldas[0]
            print('lda 0 pred:\n',lda.pred((X_val[0:10,lda.genes]).to(device)))
       print('y_val:\n',y_val[0:10])
       # Evaluate the Weighted average before NN training
       swarm_ldas.eval()
       with torch.no_grad():
           val_results=swarm_ldas.evaluate(val_loader)
           ofval_results=val_results
           print('INIT VALID LOSS: ',val_results['val_loss'])
           print('INIT VALID ACCURACY: ',val_results['val_acc'])
           if ofval_loader:
               ofval_results=swarm_ldas.evaluate(ofval_loader)
               print('INIT OFVALID LOSS: ',ofval_results['val_loss'])
               print('INIT OFVALID ACCURACY: ',ofval_results['val_acc'])
           ofval_results['random_seed']=args.random_seed
           ofval_results['numLDAs']=args.numldas
           ofval_results['incrEpochs']=0
           ofval_results['ifval_loss']=val_results['val_loss']
           ofval_results['ifval_acc']=val_results['val_acc']

           val_results=swarm_ldas.evaluate(train_loader)
           ofval_results['tr_loss']=val_results['val_loss']
           ofval_results['tr_acc']=val_results['val_acc']

           ofval_history.append(ofval_results)
           best_val_loss=ofval_results['val_loss']
           best_val_acc=ofval_results['val_acc']
       scheduler = None
       best_model=copy.deepcopy(swarm_ldas)

    if args.epochs > 0:
        start_time=time.time()
        new_history,newtr_history,newof_history,best_model,best_val_loss,best_val_acc,scheduler=train_swarmlda(swarm_ldas,
                train_loader, val_loader, ofval_loader, optimizer=optimizers[args.optimizer], n_epochs=args.epochs, 
                lr=args.lr, device=device, log_every=args.log_every, best_model=best_model, 
                best_val_loss=best_val_loss, best_val_acc=best_val_acc,scheduler=scheduler,T_0=args.t_zero,T_mult=args.t_mult,
                verbose=args.verbose)
        print(f"Training took: {time.time()-start_time} seconds")
        loss_history=loss_history+new_history
        ofloss_history=ofloss_history+newof_history
        trloss_history=trloss_history+newtr_history

    res_df=pd.DataFrame(loss_history)
    ofres_df=pd.DataFrame(ofloss_history)
    trres_df=pd.DataFrame(trloss_history)
    print(trres_df.head())
    if args.epochs > 0:
        swarm_ldas.eval()
        with torch.no_grad():
            val_results=swarm_ldas.evaluate(val_loader)
            ofval_results=val_results
            print('LAST VALID LOSS: ',val_results['val_loss'])
            print('LAST VALID ACCURACY: ',val_results['val_acc'])
            if ofval_loader:
                ofval_results=swarm_ldas.evaluate(ofval_loader)
                print('LAST OFVALID LOSS: ',ofval_results['val_loss'])
                print('LAST OFVALID ACCURACY: ',ofval_results['val_acc'])
            ofval_results['random_seed']=args.random_seed
            ofval_results['numLDAs']=args.numldas
            ofval_results['incrEpochs']=args.epochs
            ofval_results['ifval_loss']=val_results['val_loss']
            ofval_results['ifval_acc']=val_results['val_acc']
            val_results=swarm_ldas.evaluate(train_loader)
            ofval_results['tr_loss']=val_results['val_loss']
            ofval_results['tr_acc']=val_results['val_acc']
        #    sys.stdout.flush()
        ofval_history.append(ofval_results)
        # Add best_model evaluation results for us to see and compare
        best_model.eval()
        bm_results={}
        with torch.no_grad():
            bm_val_results=best_model.evaluate(val_loader)
            if ofval_loader:
                bm_ofval_results=best_model.evaluate(ofval_loader)
            else:
                bm_ofval_results=copy.deepcopy(bm_val_results)
        print('BEST MODEL LOSS: ',bm_ofval_results['val_loss'])
        print('BEST MODEL ACCURACY: ',bm_ofval_results['val_acc'])
        bm_results['random_seed']=args.random_seed
        bm_results['numLDAs']=args.numldas
        bm_results['val_loss']=bm_ofval_results['val_loss']
        bm_results['val_acc']=bm_ofval_results['val_acc']
        bm_results['ifval_loss']=bm_val_results['val_loss']
        bm_results['ifval_acc']=bm_val_results['val_acc']
        bm_results['incrEpochs']=-1
        with torch.no_grad():
            bm_val_results=best_model.evaluate(train_loader)
        bm_results['tr_loss']=bm_val_results['val_loss']
        bm_results['tr_acc']=bm_val_results['val_acc']
        ofval_history.append(bm_results)
        ofval_df=pd.DataFrame(ofval_history)
        ofval_df.to_csv(f'ValResults_{runID}.csv',index=True,index_label='runID')
        with bz2.open('ckpt_'+runID+'.bz2','wb') as fh:
            print('Saving last model ...')
            torch.save(swarm_ldas,'ckpt_'+runID+'_last.pt')
            print('Saving best model ...')
            torch.save(best_model,'ckpt_'+runID+'_best.pt')
            print('Saving initial model ...')
            torch.save(init_swarm_ldas,'ckpt_'+runID+'_init.pt')
            print('Saving the rest of variables ...')
            pickle.dump([train_loader, val_loader, ofval_loader, loss_history, trloss_history,
                         ofloss_history, res_df,trres_df, ofval_history, ofval_df,scheduler,
                         best_val_loss,best_val_acc,le], fh)
    else:
        #if ofval_loader:
        ofval_df=pd.DataFrame(ofval_history)
        #else:
        #    ofval_df=pd.DataFrame(loss_history)
        print('validation Results:\n',ofval_df)
    ##
    ## in fit validation loss
    if (args.epochs >0 or not args.test_data.strip()):
        plt.figure(figsize=(6,6))
        res_df['val_loss'].plot(style='k--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('val_loss_'+runID+'.png')
        #plt.show()
    
        plt.figure(figsize=(6,6))
        res_df['val_acc'].plot(style='k--')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig('val_acc_'+runID+'.png')
        #plt.show()
        ## Out of fit validation loss
        #plt.figure(figsize=(6,6))
        #ofres_df['val_loss'].plot(style='k--')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.savefig('ofval_loss_'+runID+'.png')
        #plt.show()
    
    ## Training loss
    plt.figure(figsize=(6,6))
    trres_df['tr_loss'].plot(style='k--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('tr_loss_'+runID+'.png')
    ## Training accuracy
    plt.figure(figsize=(6,6))
    trres_df['tr_acc'].plot(style='k--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('tr_acc_'+runID+'.png')
    #plt.show()

    deep=False
    drawLDAs=args.drawLDAs
    if (not deep) and (args.resX*args.resY == args.numldas) and drawLDAs and swarm_ldas.is_local:
        for i in range(swarm_ldas.num_of_celltypes):
            plt.figure()
            plt.imshow(swarm_ldas.lda_aggregation[i].weight.data.reshape(args.resX,args.resY).cpu().numpy())
            #plt.imshow(swarm_ldas.lda_aggregation.weight.data.reshape(10,10).cpu().numpy())
            #plt.colorbar()
            plt.savefig(f"ldas_weights_celltype{i}_{runID}.png")
        #plt.show()
        #generate_confusion_matrix(best_model, [train_loader,val_loader], le.classes_, filename='bm_confusion_matrix.png'):
        #generate_confusion_matrix(swarm_ldas, [train_loader,val_loader], le.classes_, filename='confusion_matrix.png'):
    # ## Testing with the data dataset2/tm.test_400_refined.rds
    if (args.test_data.strip()):
        testingfile=args.test_data
        if testingfile.endswith('.rds'):
            readRDS = robjects.r['readRDS']
            df_test = readRDS(testingfile)
            df_test = pandas2ri.rpy2py_dataframe(df_test)
            with open(testingfile.replace('.rds','.pkl'),'wb') as picklefile:
                pickle.dump(df_test,picklefile)
        else: #assume pickle is provided
            with open(testingfile, 'rb') as pklf:
                df_test=pickle.load(pklf)
        if (df is None): 
            with open(trainingfile, 'rb') as pklf:
                df=pickle.load(pklf)
        if (le is None):
            le = preprocessing.LabelEncoder()
            le.fit(df.yy)
            df['yy_int'] = le.transform(df.yy)
        else:
            df['yy_int'] = le.transform(df.yy)
        ##
        # Added as the numeric column in dataset7 needs to be converted to numeric.
        #
        df_test['yy']=df_test['yy'].astype(df.dtypes['yy'])
        
        # Get all the classes from both training and test data.
        test_classes=df_test['yy'].unique()
        
        all_classes = np.union1d(le.classes_,test_classes)
        ## Deal with known classes first
        print('df_test.shape:\n',df_test.shape)
        orig_df_test=df_test.copy(deep=True)
        df_test = df_test[df_test.yy.isin(le.classes_)]
        df_test['yy_int'] = le.transform(df_test.yy)
        print('df_test.shape:\n',df_test.shape)
        print('df.shape:\n',df.shape)
        print('Are the genes of test and train data the same: ',np.all(df_test.columns==df.columns))
        print('Training classes: ',df.yy.unique(),le.classes_)
        print('Test classes: ',df_test.yy.unique())
        print('df_test.shape:\n',df_test.shape)
        ## Deal with unknown classes second - -1 is used for unknown labels
        orig_df_test.loc[orig_df_test.yy.isin(le.classes_),'yy_int']=df_test['yy_int']
        orig_df_test.loc[~orig_df_test.yy.isin(le.classes_),'yy_int']=-1
        orig_X=orig_df_test.loc[:,~orig_df_test.columns.isin(['yy','yy_int'])].values.astype(np.float32)
        orig_y=orig_df_test.yy_int.values.astype(np.int64)
        orig_X,orig_y=torch.from_numpy(orig_X),torch.from_numpy(orig_y)


        testX,testy=df_test.loc[:,~df_test.columns.isin(['yy','yy_int'])].values.astype(np.float32),df_test.yy_int.values.astype(np.int64)
        X_test,y_test=torch.from_numpy(testX),torch.from_numpy(testy)
        print('Query classes:\n',torch.unique(y_test, sorted=True,return_counts=True))
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=X_test.shape[0],shuffle=False)
        # set the swarm_ldas to the best_model during training
        # add time
        start_time=time.time()
        swarm_ldas.eval()
        best_model.eval()
        init_swarm_ldas.eval()
        with torch.no_grad():
            test_results=swarm_ldas.evaluate(test_loader)
            init_test_results=init_swarm_ldas.evaluate(test_loader)
            best_test_results=best_model.evaluate(test_loader)
            val_results=best_model.evaluate(val_loader)
            if ofval_loader:
                ofval_results=best_model.evaluate(ofval_loader)
            else:
                ofval_results=copy.deepcopy(val_results)
        print(f"Query took: {time.time()-start_time} seconds")
        print('FINAL TEST LOSS - LAST: ',test_results['val_loss'])
        print('FINAL TEST ACCURACY - LAST: ',test_results['val_acc'])
        print('FINAL TEST LOSS - BEST:',best_test_results['val_loss'])
        print('FINAL TEST ACCURACY - BEST: ',best_test_results['val_acc'])
        print('FINAL TEST LOSS - INIT: ',init_test_results['val_loss'])
        print('FINAL TEST ACCURACY - INIT: ',init_test_results['val_acc'])
        test_results['random_seed']=[args.random_seed]*3
        test_results['numLDAs']=[args.numldas]*3
        test_results['ofval_loss']=[ofval_df['val_loss'].iat[-2],ofval_results['val_loss'],ofval_df['val_loss'].iat[0]]
        test_results['ofval_acc']=[ofval_df['val_acc'].iat[-2],ofval_results['val_acc'],ofval_df['val_acc'].iat[0]]
        test_results['ifval_loss']=[ofval_df['ifval_loss'].iat[-2],val_results['val_loss'],ofval_df['ifval_loss'].iat[0]]
        test_results['ifval_acc']=[ofval_df['ifval_acc'].iat[-2],val_results['val_acc'],ofval_df['ifval_acc'].iat[0]]
        test_results['val_loss']=[test_results['val_loss'],best_test_results['val_loss'],init_test_results['val_loss']]
        test_results['val_acc']=[test_results['val_acc'],best_test_results['val_acc'],init_test_results['val_acc']]

        
        testres_df=pd.DataFrame(test_results)
        testres_df.to_csv(f'QueryResults_{runID}.csv',index=False)
        # Last model
        our_output=swarm_ldas.pred(X_test.to(device)).reshape(-1).cpu().numpy()

        with torch.no_grad():
            num_of_cell_types=len(le.classes_)
            uniform_prob=1/num_of_cell_types*torch.ones(num_of_cell_types,device=device)
            cum_uniform=torch.cumsum(uniform_prob,0)
            res_unknwn_df=orig_df_test[['yy','yy_int']].copy()
            #for X_test in orig_X.split(batch_size): we might need to batch this if query cannot fit on GPU
            our_out_probs=swarm_ldas.pred_prob(orig_X.to(device))
            our_out_cumprobs=our_out_probs.cumsum(axis=1) #reshape(-1).cpu().numpy()
            res_unknwn_df.loc[:,'yy_pred']= torch.argmax(our_out_probs, axis=1).reshape(-1).cpu().numpy()

            # compute the confidence we have by computing Wasserstein distance to uniform probability.
            # KL divergence metrics and Wasserstein distnace.
            
            kl_divergence=(our_out_probs*torch.log(our_out_probs/uniform_prob+1e-10)).sum(axis=1).cpu().numpy()
            ws_distance= (our_out_cumprobs-cum_uniform).abs().sum(axis=1).cpu().numpy()
            res_unknwn_df.loc[:,le.classes_]=our_out_probs.cpu().numpy()
            res_unknwn_df.loc[:,'KL']=kl_divergence
            res_unknwn_df.loc[:,'WS']=ws_distance

            res_unknwn_df.to_csv(f'QueryReportLastUnknown_{runID}.csv',index=False)

        actual_output=df_test.yy_int.values
        
        #num_of_cell_types=len(le.classes_)
        print('Miss-classified output Last:\n',our_output[our_output!=actual_output])
        print('Correct output Last:\n',actual_output[our_output!=actual_output])
        print('Verified Accuracy Last: ',len(actual_output[our_output==actual_output])/len(actual_output))
        #print('Verified Accuracy Last: ',len(actual_output[our_output==actual_output])*prob(our_output)/len(actual_output))
        #print('Verified Accuracy Last: ',len(actual_output[our_output!=actual_output])*(1-prob(our_output))/len(actual_output))
        our_output_label=le.inverse_transform(our_output)
        actual_output_label=le.inverse_transform(actual_output)
        cf_matrix = confusion_matrix(actual_output_label,our_output_label,labels=le.classes_)
        pd.DataFrame(cf_matrix,columns=le.classes_,index=le.classes_).to_csv(f'QueryConfusionMatrixLast_{runID}.csv')
        pd.DataFrame(classification_report(actual_output_label,our_output_label,labels=range(num_of_cell_types),target_names=le.classes_,output_dict=True)).to_csv(f'QueryReportLast_{runID}.csv',index=False)
            
        # Best model
        our_output=best_model.pred(X_test.to(device)).reshape(-1).cpu().numpy()
        with torch.no_grad():
            num_of_cell_types=len(le.classes_)
            uniform_prob=1/num_of_cell_types*torch.ones(num_of_cell_types,device=device)
            cum_uniform=torch.cumsum(uniform_prob,0)
            res_unknwn_df=orig_df_test[['yy','yy_int']].copy()
            #for X_test in orig_X.split(batch_size): we might need to batch this if query cannot fit on GPU
            our_out_probs=best_model.pred_prob(orig_X.to(device))
            our_out_cumprobs=our_out_probs.cumsum(axis=1) #reshape(-1).cpu().numpy()
            res_unknwn_df.loc[:,'yy_pred']= torch.argmax(our_out_probs, axis=1).reshape(-1).cpu().numpy()
            
            print(f'our_out_probs device: {our_out_probs.get_device()}')
            # compute the confidence we have by computing Wasserstein distance to uniform probability.
            # KL divergence metrics and Wasserstein distnace.
            
            print(f'uniform device: {uniform_prob.get_device()}')
            kl_divergence=(our_out_probs*torch.log(our_out_probs/uniform_prob+1e-10)).sum(axis=1).cpu().numpy()
            ws_distance= (our_out_cumprobs-cum_uniform).abs().sum(axis=1).cpu().numpy()
            res_unknwn_df.loc[:,le.classes_]=our_out_probs.cpu().numpy()
            res_unknwn_df.loc[:,'KL']=kl_divergence
            res_unknwn_df.loc[:,'WS']=ws_distance
            res_unknwn_df.to_csv(f'QueryReportBestUnknown_{runID}.csv',index=False)

        print('Miss-classified output best:\n',our_output[our_output!=actual_output])
        print('Correct output best:\n',actual_output[our_output!=actual_output])
        print('Verified Accuracy best: ',len(actual_output[our_output==actual_output])/len(actual_output))
        our_output_label=le.inverse_transform(our_output)
        actual_output_label=le.inverse_transform(actual_output)
        cf_matrix = confusion_matrix(actual_output_label,our_output_label,labels=le.classes_)
        pd.DataFrame(cf_matrix,columns=le.classes_,index=le.classes_).to_csv(f'QueryConfusionMatrixBest_{runID}.csv')
        pd.DataFrame(classification_report(actual_output_label,our_output_label,labels=range(num_of_cell_types),target_names=le.classes_,output_dict=True)).to_csv(f'QueryReportBest_{runID}.csv',index=False)
 
        # Init model
        our_output=init_swarm_ldas.pred(X_test.to(device)).reshape(-1).cpu().numpy()
        with torch.no_grad():
            num_of_cell_types=len(le.classes_)
            uniform_prob=1/num_of_cell_types*torch.ones(num_of_cell_types,device=device)
            cum_uniform=torch.cumsum(uniform_prob,0)
            res_unknwn_df=orig_df_test[['yy','yy_int']].copy()
            #for X_test in orig_X.split(batch_size): we might need to batch this if query cannot fit on GPU
            our_out_probs=init_swarm_ldas.pred_prob(orig_X.to(device))
            our_out_cumprobs=our_out_probs.cumsum(axis=1) #reshape(-1).cpu().numpy()
            res_unknwn_df.loc[:,'yy_pred']= torch.argmax(our_out_probs, axis=1).reshape(-1).cpu().numpy()
            
            print(f'our_out_probs device: {our_out_probs.get_device()}')
            # compute the confidence we have by computing Wasserstein distance to uniform probability.
            # KL divergence metrics and Wasserstein distnace.
            
            print(f'uniform device: {uniform_prob.get_device()}')
            kl_divergence=(our_out_probs*torch.log(our_out_probs/uniform_prob+1e-10)).sum(axis=1).cpu().numpy()
            ws_distance= (our_out_cumprobs-cum_uniform).abs().sum(axis=1).cpu().numpy()
            res_unknwn_df.loc[:,le.classes_]=our_out_probs.cpu().numpy()
            res_unknwn_df.loc[:,'KL']=kl_divergence
            res_unknwn_df.loc[:,'WS']=ws_distance
            res_unknwn_df.to_csv(f'QueryReportInitUnknown_{runID}.csv',index=False)
        
        print('Miss-classified output init:\n',our_output[our_output!=actual_output])
        print('Correct output init:\n',actual_output[our_output!=actual_output])
        print('Verified Accuracy init: ',len(actual_output[our_output==actual_output])/len(actual_output))
        our_output_label=le.inverse_transform(our_output)
        actual_output_label=le.inverse_transform(actual_output)
        cf_matrix = confusion_matrix(actual_output_label,our_output_label,labels=le.classes_)
        pd.DataFrame(cf_matrix,columns=le.classes_,index=le.classes_).to_csv(f'QueryConfusionMatrixInit_{runID}.csv')
        pd.DataFrame(classification_report(actual_output_label,our_output_label,labels=range(num_of_cell_types),target_names=le.classes_,output_dict=True)).to_csv(f'QueryReportInit_{runID}.csv',index=False)
        


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    run(args)
