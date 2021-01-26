import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import confusion_matrix
import math

def smote_over_sampling(t_train, x_train,target_class):

    os = SMOTE(random_state=0)

    columns = x_train.columns

    os_data_x,os_data_t=os.fit_sample(x_train, t_train)
    os_data_x = pd.DataFrame(data=os_data_x,columns=columns )
    os_data_t= pd.DataFrame(data=os_data_t,columns=[target_class])

    # check the numbers of our data
    print("length of oversampled data is ",len(os_data_x))
    print("Number of no "+ target_class +" in oversampled data",len(os_data_t[os_data_t[target_class]==0]))
    print("Number of "+ target_class+" in oversampled data",len(os_data_t[os_data_t[target_class]==1]))

    return os_data_x,os_data_t

def pred_model(log_reg,x_test ,t_test,target_class):

    if len(x_test) == 0:
        return 'Null'

    # get the confusion matrix and classification report on test
    pred_target_test = log_reg.predict(x_test)

    t_prob=log_reg.predict_proba(x_test)[:, 1] #positive class prediction probabilities
    x_test['t_prob'] = t_prob

    x_test['pred_'+target_class] = pred_target_test

    x_test[target_class]=t_test

    return x_test


def creat_model(x_train,t_train):

    log_reg = LogisticRegression()

    create_log_reg=log_reg.fit(x_train, t_train)

    return create_log_reg

def save_confusion_matrix(results,target_class,confusion_matrix_file):
    tn, fp, fn, tp = confusion_matrix(results[target_class], results['pred_' + target_class]).ravel()
    FPR = (float(fp)/ float(tn + fp))
    TPR_Recall = (float(tp)/ float(tp + fn))


    accuracy=(float(tp+tn)/ float(tp+fp+tn+fn))
    specificity=(float(tn)/ float(fp+tn))
    precision=(float(tp)/ float(tp+fp))
    x=(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    MCC=(float(tp*tn)-float(fp*fn))/math.sqrt(x)


    file_fp_tp = open(confusion_matrix_file, 'w')
    file_fp_tp.write("the tn: " + str(tn))
    file_fp_tp.write('\n')
    file_fp_tp.write("the fp: " + str(fp))
    file_fp_tp.write('\n')
    file_fp_tp.write("the fn: " + str(fn))
    file_fp_tp.write('\n')
    file_fp_tp.write("the tp: " + str(tp))
    file_fp_tp.write('\n')
    file_fp_tp.write("the FPR: " + str(FPR))
    file_fp_tp.write('\n')
    file_fp_tp.write("the TPR: " + str(TPR_Recall))
    file_fp_tp.write('\n')
    file_fp_tp.write("the accuracy: " + str(accuracy))
    file_fp_tp.write('\n')
    file_fp_tp.write("the specificity: " + str(specificity))
    file_fp_tp.write('\n')
    file_fp_tp.write("the precision: " + str(precision))
    file_fp_tp.write('\n')
    file_fp_tp.write("the MCC: " + str(MCC))
    file_fp_tp.write('\n')
    file_fp_tp.close()

def main(t_train, x_train, t_test, x_test,target_class,validaion_or_test,DIRECTORY):

    DIRECTORY = DIRECTORY + r'/Logistic_Regression'
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    if validaion_or_test:
        DIRECTORY = DIRECTORY +r'/validation'
    else:
        DIRECTORY = DIRECTORY + r'/test'

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    result_file = DIRECTORY + '\\logistic_regression.csv'
    confusion_matrix_file = DIRECTORY + r'\\confusion_matrix_log_reg.txt'

    if len(x_train) == 0:
        print ("Empty data frame")
        return 'Null'


    os_x_train, os_t_train = smote_over_sampling(t_train, x_train, target_class)# deling with the unbalanced data

    log_reg = creat_model(os_x_train,os_t_train)

    if log_reg != 'Null':
        results=pred_model(log_reg,x_test ,t_test,target_class)
        results.to_csv(result_file, sep=',', index=False)
        save_confusion_matrix(results, target_class, confusion_matrix_file)

    return results,log_reg

if __name__ == '__main__':
    main()
