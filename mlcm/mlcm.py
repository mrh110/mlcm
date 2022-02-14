'''
    Please read the following paper for more information:\
    M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix, 
    IEEE Access, Feb. 2022, DOI: 10.1109/ACCESS.2022.3151048
'''

import numpy as np

def cm(label_true,label_pred,print_note=True):
    '''
    Computes the "Multi-Lable Confusion Matrix" (MLCM). 
    MLCM satisfies the requirements of a 2-dimensional confusion matrix.
    Please read the following paper for more information:\
    M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix,
    IEEE Access, Feb. 2022, DOI: 10.1109/ACCESS.2022.3151048
    
    Parameters
    ----------
    label_true : {arraylike, sparse matrix} of shape (num_instance,num_classes)
        Assigned (True) labels in one-hot-encoding format.
        
    label_pred : {arraylike, sparse matrix} of shape (num_instance,num_classes)
        Predicted labels in one-hot-encoding format.
        
    print_note : bool, default=True
        If true, shows a note on the dimension of the confusion matrix.

    Returns
    -------
    conf_mat: multi-label confusion matrix (MLCM)
        ndarray of shape (num_classes+1, num_classes+1).
        Rows represent True labels and columns represent Predicted labels.
        The last row is for "No True Label" assigned (NTL).
        The last column is for "No Predicted Label" found (NPL).
        
    normal_conf_mat: normalized multi-label confusion matrix (normalizd MLCM)
        Numbers show the percentage.
        
    Notes
    -----
    Implemented by Mohammadreza Heydarian, at BioMedic.AI (McMaster University)
    Aug 13, 2020; Modified: Feb 8, 2022.
    '''

    num_classes = len(label_pred[0])  # number of all classes
    num_instances = len(label_pred)  # number of instances (input) 
    # initializing the confusion matrix
    conf_mat = np.zeros((num_classes+1,num_classes+1), dtype=np.int64) 

    for i in range(num_instances): 

        num_of_true_labels = np.sum(label_true[i])
        num_of_pred_labels = np.sum(label_pred[i])

        if num_of_true_labels == 0: 
            if num_of_pred_labels == 0: 
                conf_mat[num_classes][num_classes] += 1 
            else:
                for k in range(num_classes):
                    if label_pred[i][k] == 1:  
                        conf_mat[num_classes][k] += 1  # NTL 


        elif num_of_true_labels == 1:  
            for j in range(num_classes): 
                if label_true[i][j] == 1:  
                    if num_of_pred_labels == 0: 
                        conf_mat[j][num_classes] += 1  # NPL 
                    else: 
                        for k in range(num_classes): 
                            if label_pred[i][k] == 1:  
                                conf_mat[j][k] += 1 

        else: 
            if num_of_pred_labels == 0: 
                for j in range(num_classes): 
                    if label_true[i][j] == 1: 
                        conf_mat[j][num_classes] += 1  # NPL               
            else: 
                true_checked = np.zeros((num_classes,1), dtype=np.int) 
                pred_checked = np.zeros((num_classes,1), dtype=np.int) 
                # Check for correct prediction
                for j in range(num_classes): 
                    if label_true[i][j] == 1: 
                        if label_pred[i][j] == 1: 
                            conf_mat[j][j] += 1 
                            true_checked[j] = 1 
                            pred_checked[j] = 1  
                # check for incorrect prediction(s)
                for k in range(num_classes): 
                    if (label_pred[i][k] == 1) and (pred_checked[k] != 1): 
                        for j in range(num_classes):
                            if (label_true[i][j] == 1)and(true_checked[j]!=1):
                                conf_mat[j][k] += 1 
                                pred_checked[k] = 1 
                                true_checked[j] = 1 
                # check for incorrect prediction(s) while all True labels were
                # predicted correctly
                for k in range(num_classes):
                    if (label_pred[i][k] == 1) and (pred_checked[k] != 1): 
                        for j in range(num_classes): 
                            if (label_true[i][j] == 1): 
                                conf_mat[j][k] += 1 
                                pred_checked[k] = 1 
                                true_checked[j] = 1 
                # check for cases with True label(s) and no predicted label
                for k in range(num_classes):
                    if (label_true[i][k] == 1) and (true_checked[k] != 1): 
                        conf_mat[k][num_classes] += 1  # NPL               

    # calculating the normal confusion matrix
    divide = conf_mat.sum(axis=1, dtype='int64') 
    for indx in range(len(divide)):
        if divide[indx] == 0:  # To avoid division by zero
            divide[indx] = 1

    normal_conf_mat = np.zeros((len(divide),len(divide)), dtype=np.float64)
    for i in range (len(divide)):
        for j in range (len(divide)):
            normal_conf_mat[i][j] = round((float(conf_mat[i][j]) / divide[i]) \
                                          *100)

    if print_note:
        print('MLCM has one extra row (NTL) and one extra column (NPL).\
        \nPlease read the following paper for more information:\n\
        Heydarian et al., MLCM: Multi-Label Confusion Matrix, IEEE Access,2022\
        \nTo skip this message, please add parameter "print_note=False"\n\
        e.g., conf_mat,normal_conf_mat = mlcm.cm(label_true,label_pred,False)')

    return conf_mat, normal_conf_mat


# ########################################################################### #
# Developing a function to produce some statistics based on the MLCM  
# ########################################################################### #
def stats(conf_mat, print_binary_mat=True):
    '''
    Computes one-vs-rest confusion matrices for all classes of multi-label data
    The One-vs-rest is a 2x2 matrix contains [TN,FP; FN,TP]
    Prints precision, recall, and f1-score for each class.
    Prints micro, macro, and weighted average of precision, recall, and
    F1-score over all classes.
    
    Parameters
    ----------
    conf_mat : multi-label confusion matrix (MLCM)
        ndarray of shape (num_classes+1, num_classes+1).
        Rows represent True labels and columns represent Predicted labels.
        The last row is for "No True Label" assigned (NTL).
        The last column is for "No Predicted Label" found (NPL).
        
    print_binary_mat : bool, default=True
        If true, prints all one-vs-rest confusion matrices.
        
    Returns
    -------
    one_vs_rest : a set of one vs rest (binary) confusion matrices, one for
                 each class
    
    
    prints
    ------
    precision, recall, F1-score, and weight for each of classes, also micro, 
    macro, and weighted average of precision, recall, and F1-score
    over all classes.

    Notes
    -----
    For the cases that all instances in the dataset have at least one label, 
    all cells in the last row of MLCM are zero and are ignored for calculating 
    the statistics, but the information in the last column of MLCM are involved 
    in calculating the statistics.

    Implemented by Mohammadreza Heydarian, at BioMedic.AI (McMaster University)
    Aug 13, 2020; Modified: Feb 8, 2022.
    '''
    num_classes = conf_mat.shape[1]
    tp = np.zeros(num_classes, dtype=np.int64)  
    tn = np.zeros(num_classes, dtype=np.int64)  
    fp = np.zeros(num_classes, dtype=np.int64)  
    fn = np.zeros(num_classes, dtype=np.int64)  

    precision = np.zeros(num_classes, dtype=np.float)  
    recall = np.zeros(num_classes, dtype=np.float)  
    f1_score = np.zeros(num_classes, dtype=np.float)  

    # Calculating TP, TN, FP, FN from MLCM
    for k in range(num_classes): 
        tp[k] = conf_mat[k][k]
        for i in range(num_classes):
            if i != k:
                tn[k] += conf_mat[i][i]
                fp[k] += conf_mat[i][k]
                fn[k] += conf_mat[k][i]

    # Creating one_vs_rest confusion matrices
    one_vs_rest = np.zeros((num_classes,2,2), dtype='int64')
    one_vs_rest [:,0,0] = tn
    one_vs_rest [:,0,1] = fp
    one_vs_rest [:,1,0] = fn
    one_vs_rest [:,1,1] = tp
    if print_binary_mat:
        print(one_vs_rest)

    # Calculating precision, recall, and F1-score for each of classes
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*tp/(2*tp+fn+fp)

    divide = conf_mat.sum(axis=1, dtype='int64') # sum of each row of MLCM

    if divide[-1] != 0: # some instances have not been assigned with any label 
        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

        print('\n       class#     precision        recall      f1-score\
        weight\n')
        sp = '        '
        sp2 = '  '
        total_weight = divide.sum()
        float_formatter = "{:.2f}".format
        for k in range(num_classes-1):
            print(sp2,sp,k,sp,float_formatter(precision[k]),sp, \
                  float_formatter(recall[k]), sp,float_formatter(f1_score[k]),\
                  sp,divide[k])
        k = num_classes-1
        print(sp,' NTL',sp,float_formatter(precision[k]),sp, \
              float_formatter(recall[k]), sp,float_formatter(f1_score[k]), \
              sp,divide[k])

        print('\n    micro avg',sp,float_formatter(micro_precision),sp, \
              float_formatter(micro_recall),sp,float_formatter(micro_f1),\
              sp,total_weight)
        print('    macro avg',sp,float_formatter(macro_precision),sp,
              float_formatter(macro_recall),sp,float_formatter(macro_f1),sp,\
              total_weight)
        print(' weighted avg',sp,float_formatter(weighted_precision),sp,\
              float_formatter(weighted_recall),sp, \
              float_formatter(weighted_f1),sp,total_weight)
    else:
        precision = precision[:-1]
        recall = recall[:-1]
        f1_score = f1_score[:-1]
        divide = divide[:-1]
        num_classes -= 1

        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

        print('\n       class#     precision        recall      f1-score\
        weight\n')
        sp = '        '
        sp2 = '  '
        total_weight = divide.sum()
        float_formatter = "{:.2f}".format
        for k in range(num_classes):
            print(sp2,sp,k,sp,float_formatter(precision[k]),sp, \
                  float_formatter(recall[k]), sp,\
                  float_formatter(f1_score[k]),sp,divide[k])
        print(sp,' NoC',sp,'There is not any data with no true-label assigned!')

        print('\n    micro avg',sp,float_formatter(micro_precision),sp,\
              float_formatter(micro_recall),sp,float_formatter(micro_f1),\
              sp,total_weight)
        print('    macro avg',sp,float_formatter(macro_precision),sp,\
              float_formatter(macro_recall),sp,float_formatter(macro_f1),sp,\
              total_weight)
        print(' weighted avg',sp,float_formatter(weighted_precision),sp,\
              float_formatter(weighted_recall),sp,\
              float_formatter(weighted_f1),sp,total_weight)

    return one_vs_rest
