import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import math



















def creat_model(x_train,t_train):


    rf = RandomForestClassifier(n_estimators=100, min_samples_split=5,min_samples_leaf=5,criterion='entropy',
                                oob_score='true', class_weight='balanced', max_features='sqrt',random_state=0)


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=5)]
    # Create the grid
    param_grid = {'n_estimators': n_estimators}
    rf_grid_search = GridSearchCV(rf, param_grid=param_grid,cv=3)

    create_random_forest = rf_grid_search.fit(x_train, t_train)

    print (create_random_forest.best_params_)


    print ("the oob_score is: "+str(create_random_forest.best_estimator_.oob_score_))

    return create_random_forest


def save_obj(name, obj):
    path = name + ".pkl"
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def pred_model(rf,x_test ,t_test,target_class):

    rf=rf.best_estimator_

    if len(x_test) == 0:
        return 'Null'

    # get the confusion matrix and classification report on test
    pred_target_test = rf.predict(x_test)

    x_test['t_prob'] = rf.predict_proba(x_test)[:, 1]

    x_test['pred_'+target_class] = pred_target_test

    x_test[target_class]=t_test

    return x_test

def save_features_importance(random_forest,feature_cols,DIRECTORY):

    fi1=random_forest.best_estimator_.feature_importances_



    fim = []
    ind = 0
    for f in feature_cols:
        fim.append([fi1[ind], f])
        ind += 1

    fim = sorted(fim)
    #print (fim)

    df_feature_cols = pd.DataFrame(fim, columns=["importance", "feature"])
    df_feature_cols.to_csv(DIRECTORY + r'\\importance_of_features.csv', index=False)

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

    DIRECTORY = DIRECTORY + r'/Random_Forest'
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    if validaion_or_test:
        DIRECTORY = DIRECTORY +r'/validation'
    else:
        DIRECTORY = DIRECTORY + r'/test'

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    result_file = DIRECTORY + '\\results_random_forest.csv'
    tree_name = DIRECTORY + '\\forest'
    confusion_matrix_file=DIRECTORY + r'\\confusion_matrix_RF.txt'



    if len(x_train) == 0:
        print ("Empty data frame")
        return 'Null'

    random_forest = creat_model(x_train,t_train)
    feature_cols = [col for col in x_train.columns]
    if random_forest != 'Null':
        save_obj(tree_name, random_forest)
        save_features_importance(random_forest,feature_cols,DIRECTORY)
        results=pred_model(random_forest,x_test ,t_test,target_class)
        results.to_csv(result_file, sep=',', index=False)
        save_confusion_matrix(results, target_class, confusion_matrix_file)

    return results,random_forest

if __name__ == '__main__':
    main()


