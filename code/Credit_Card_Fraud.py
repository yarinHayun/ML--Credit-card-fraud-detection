import pandas as pd
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
import Random_Forecast
import Logistic_Regression
import numpy as np
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

DIRECTORY= r"./"
try:
    os.mkdir(DIRECTORY+r"/FIG/")
except OSError:
    print("Creation of the directory  failed")
else:
    print("Successfully created the directory ")
DATA_FILE=r"creditcard.csv"
TARGET_CLASS= "Class"


def precision_recall_curve(classifiers_comparison,x_test):

    for c in classifiers_comparison:
        c_results=c[0]
        c_name=c[1]
        classifier=c[2]

        t_test = c_results[TARGET_CLASS]
        t_prob=c_results['t_prob']
        t_pred=c_results['pred_'+TARGET_CLASS]

        average_precision = average_precision_score(t_test, t_prob)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))

        disp = plot_precision_recall_curve(classifier, x_test, t_test)
        disp.ax_.set_title(c_name+'2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision))

        plt.savefig(DIRECTORY + r"/FIG/" + c_name+'PRC')
        plt.close()

def features_visualization(df):

    df_features = list(df.columns)
    df_features.remove(TARGET_CLASS)
    for col in df_features:
        # sns_plot=sns.distplot(df[col])#univariate distribution of each feature
        # plt.savefig(DIRECTORY+r"/FIG/"+"univariate distribution of " + col + " feature.png")
        # plt.close()
        ######################################
        sns_plot_2=sns.jointplot(x=TARGET_CLASS, y=col, data=df)  #  bivariate distributions of features pair
        plt.savefig(DIRECTORY+r"/FIG/"+"bivariate distributions of "+TARGET_CLASS+" and "+col+" pair.png")
        plt.close()


def features_discretization_in_test(df,features,quantiles):
    for f in features:
        f_discr=pd.cut(df[f], bins=len(quantiles[f]), labels=quantiles[f]) #discretize feature into equal-sized buckets
        df[f]=f_discr

    return df

def features_discretization(df,features):# using Freedman-Diaconis rule
    quantiles_f={}
    for f in features:
        Q1 = df[f].quantile(0.25)
        Q3 = df[f].quantile(0.75)
        IQR = Q3 - Q1
        h = (2 * IQR) / (len(df[f]) ** (1/3))  # select the width of the bins using Freedman-Diaconis rule
        q = int(round((max(df[f]) - min(df[f])) / h))  # number of bins
        f_discr=pd.cut(df[f],bins =q,labels=list(range(q)),retbins =True) #discretize feature into  buckets
        quantiles=list(f_discr[1])
        df[f]=f_discr[0]
        quantiles_f[f]=quantiles

    return df,quantiles_f


def visualization_for_features_correlation(df):
    # sns.set(font_scale=1.5)
    # sns.heatmap(df.corr(), square=True, cbar=True, annot=True, annot_kws={'size':3})  # correlation of features
    # plt.savefig(DIRECTORY+r"/FIG/"+'features_correlation.png')
    # plt.close()
    df.corr().to_csv(DIRECTORY+ r"features_correlation.csv", index=True)


def dealing_with_correlative_features(df,C):
    visualization_features=list(df.columns)
    visualization_for_features_correlation(df[visualization_features])
    # create correlation matrix
    corr_matrix = df.corr().abs()
    # Find index of feature columns with correlation greater than C
    indices = np.where(corr_matrix > C)
    indices = [(corr_matrix.index[x], corr_matrix.columns[y])
               for x, y in zip(*indices) if x != y and x < y]

    to_drop=set()
    for f in indices:
        f1_target_corr=abs(df[f[0]].corr(df[TARGET_CLASS]))
        f2_target_corr=abs(df[f[1]].corr(df[TARGET_CLASS]))

        if f2_target_corr>f1_target_corr:
            to_drop.add(f[0])
        else:
            to_drop.add(f[1])
    print (to_drop)
    print ("end drop of correlative_features")
    df.drop(to_drop, axis=1, inplace=True)
    return df



def visualization_for_outliers(df,features_to_cheack_outliers):

    for f in features_to_cheack_outliers:


        plt.figure(figsize=(10, 8))
        plt.subplot(211)
        plt.xlim(df[f].min(), df[f].max() * 1.1)

        ax = df[f].plot(kind='kde')

        plt.subplot(212)
        plt.xlim(df[f].min(), df[f].max() * 1.1)
        sns.boxplot(x=df[f])
        plt.savefig(DIRECTORY+r"/FIG/"+'cheack_outliers for '+f+'.png')


def main():

    try:

        df_data = pd.read_csv(DIRECTORY + DATA_FILE)

    except:
        sys.exit("[!] Can't open data file csv.")

    ############################### Data Exploration and Preparation #####################################

    print (len(df_data))
    print (len(df_data[df_data[TARGET_CLASS] == 1]))
    ##############feature selection#############
    # df_data = df_data.drop(['V22', 'V24'], axis=1)
    #################################

    df_data.info()    #print the types of features
    ##split into train and test sets
    train_0, test_0 = train_test_split(df_data[df_data[TARGET_CLASS] == 0], test_size=0.25)
    train_1, test_1 = train_test_split(df_data[df_data[TARGET_CLASS] == 1], test_size=0.25)

    df_train = pd.concat([train_0, train_1], ignore_index=True)
    df_test = pd.concat([test_0, test_1], ignore_index=True)

    print ("the num of rowes in train is " + str(len(df_train)))

    print("the count of each type in the target class is : \n" + str(
        df_train[TARGET_CLASS].value_counts()))  # check whether the data is unbalanced


    df_statistics=df_train.describe()
    print (df_statistics.describe())
    df_statistics.to_csv(DIRECTORY+ r"train_describe.csv", index=True)

    df_F_mean_per_class=df_train.groupby(TARGET_CLASS).mean()
    print(df_F_mean_per_class)
    df_F_mean_per_class.to_csv(DIRECTORY + r"F_mean_per_class.csv", index=True)


    # #visualization
    features_visualization(df_train)
    #dealing with outliers
    features_to_cheack_outliers = list(df_train.columns)
    features_to_cheack_outliers.remove(TARGET_CLASS)
    visualization_for_outliers(df_train,features_to_cheack_outliers)

    # dealing with correlative features
    df_train = dealing_with_correlative_features(df_train, 0.7)

    features_for_discretization=list(df_train.columns)
    features_for_discretization.remove(TARGET_CLASS)
    print("Count unique values for each feature BEFORE discretization:")
    print(df_train[features_for_discretization].nunique())

    # drop duplicates rows from train
    df_train.drop_duplicates()
    print (len(df_train))

    #discrete features to avoid overfitting
    # df_train, quantiles = features_discretization(df_train, features_for_discretization)
    # print("Count unique values for each feature AFTER discretization:")
    # print(df_train[features_for_discretization].nunique())
    #
    # df_train=df_train.astype(int)

    ############################### Data Modelling #####################################

    ##split into train and validation sets
    train_0, validation_0 = train_test_split(df_train[df_train[TARGET_CLASS] == 0], test_size=0.25)
    train_1, validation_1 = train_test_split(df_train[df_train[TARGET_CLASS] == 1], test_size=0.25)

    train = pd.concat([train_0, train_1], ignore_index=True)
    validation = pd.concat([validation_0, validation_1], ignore_index=True)


    t_train = train[TARGET_CLASS]
    x_train = train.drop([TARGET_CLASS], axis=1)
    t_validation = validation[TARGET_CLASS]
    x_validation = validation.drop([TARGET_CLASS], axis=1)


    logistic_regression_results_on_validation,classifier_LG = Logistic_Regression.main(t_train.copy(), x_train.copy(), t_validation.copy(), x_validation.copy(), TARGET_CLASS,True, DIRECTORY)
    print ('Done logistic_regression')
    random_forest_results_on_validation,classifier_RD= Random_Forecast.main(t_train.copy(), x_train.copy(), t_validation.copy(), x_validation.copy(), TARGET_CLASS,True,DIRECTORY)
    print ('Done random_forest')

    c1 = (logistic_regression_results_on_validation, 'lin_reg', classifier_LG)
    c2 = (random_forest_results_on_validation, 'RF', classifier_RD)

    classifiers_comparison = [c1, c2]
    precision_recall_curve(classifiers_comparison,x_validation)

    ############################### Evaluation #####################################

    # features discretization
    # df_test=features_discretization_in_test(df_test,features_for_discretization,quantiles)
    # df_test=df_test.astype(int)

    t_train = df_train[TARGET_CLASS]
    x_train = df_train.drop([TARGET_CLASS], axis=1)
    t_test = df_test[TARGET_CLASS]
    x_test = df_test.filter(items=x_train.keys())


    random_forest_results_on_test,classifier_RD_test = Random_Forecast.main(t_train.copy(), x_train.copy(), t_test.copy(),x_test.copy(), TARGET_CLASS,False, DIRECTORY)
    c3 = (random_forest_results_on_test, 'RF_test', classifier_RD_test)

    logistic_regression_results_on_test,classifier_LG_test  = Logistic_Regression.main(t_train.copy(), x_train.copy(), t_test.copy(),x_test.copy(), TARGET_CLASS,False, DIRECTORY)
    c4 = (logistic_regression_results_on_test, 'lin_reg', classifier_LG_test)

    classifiers_comparison_test = [c3,c4]
    precision_recall_curve(classifiers_comparison_test, x_test)
    print ('Done')


if __name__ == "__main__":

    main()


