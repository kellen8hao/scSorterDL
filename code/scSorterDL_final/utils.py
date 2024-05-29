#!/usr/bin/env python
# Some utility functions for scSorterDL
import numpy as np
import argparse
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay


## Argument parsing for python script command line...
def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, choices=['cuda', 'cpu','auto'],default='auto')
    parser.add_argument('--restart', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('-b','--batch_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--cellsr', type=float, default=0.80)
    parser.add_argument('--ldareg', type=float, default=0.1)
    parser.add_argument('--global_reg', action='store_true')
    parser.add_argument('--noshrinkage', action='store_true')
    parser.add_argument('--drawLDAs', action='store_true')
    parser.add_argument('--log_every', type=int, default=30)
    parser.add_argument('--resX', type=int, default=25)
    parser.add_argument('--resY', type=int, default=40)
    parser.add_argument('--numldas', type=int, default=1000)
    parser.add_argument('--gene_sampling', type=str, default='uniform',
        choices=['uniform','gene_info','dirichlet'])
    parser.add_argument('--cell_sampling', type=str, default='uniform',
        choices=['uniform','balanced','stratified', 'group_split'])

    parser.add_argument('--train_data', type=str, default='/project/def-ubcxzh/kbai/New\ data/dataset1/pancreas.training_400_refined.rds')
#    parser.add_argument('--gene_info', type=str, default='/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform/dataset1/wilcox_res/gene_select_celltype.pkl')
    parser.add_argument('--test_data', type=str, default='')
    parser.add_argument('--runID', type=str, default='pancreas_1000_1') #data set_ num of ldas _ random seed
    parser.add_argument('-o','--optimizer', type=str, default='NAdam') #Specify the optimizer ...
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('-t0', '--t-zero', dest='t_zero', type=int,
                    default=100)
    parser.add_argument('-tm', '--t-mult', dest='t_mult', type=int,
                    default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-r', '--retrain', action='store_true')
    parser.add_argument('-c', '--common_cell_sampler', action='store_true')
    parser.add_argument('--fine_aggr', action='store_true')
    parser.add_argument('--web_layer_sizes', nargs='*', type=positive) # if --not 
    args = parser.parse_args(arguments)
    return args

def positive(value):
    
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def generate_confusion_matrix(model, data_loaders, classes, filename='confusion_matrix.png'):
    pred_output=np.empty(0)
    true_output=np.empty(0)

    for data_loader in data_loaders:
        for X,y in data_loader:
            pred_output=np.append(pred_output,model.pred(X.to(device)).reshape(-1).cpu().numpy())
            true_output=np.append(true_output,y)
    cf_matrix = confusion_matrix(true_output, pred_output)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    #sn.heatmap(df_cm, annot=True)
    ConfusionMatrixDisplay.from_predictions(true_output,pred_output,labels=classes)
    plt.savefig(filename)
    return df_cm,classification_report(true_output,pred_output,target_names=classes)

