import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns

def data():
    columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'ghclass']
    file = 'magic04.data'
    dataset = pd.read_csv(file, names=columns, sep=',')
    feature = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
    #dataset['ghclass'][dataset['ghclass'] == 'g'] = 1
    #dataset['ghclass'][dataset['ghclass'] == 'h'] = 0
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 10].values
    print(dataset)
    sns.pairplot(dataset.dropna(), vars=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'], hue='ghclass', palette="Set2");
    return (X,y)

def kNNClassifier(X, y, X_train, X_test, y_train, y_test, cvValue, k_range):
    print("------------KNN CLASSIFIER------------")
    kNNaccr_scores=np.empty((40))
    kNNcv_accr_scores_mean = []
    kNNcv_scores_std = [] #standart derivation
    kNNcv_neglog_std = []
    kNNcv_rocauc_std = []
    kNNcv_rocauc_scores_mean = []
    kNNcv_neglog_scores_mean = []
    for k in k_range:
        kNNclf = KNeighborsClassifier(n_neighbors=k, weights="uniform", algorithm="auto")
        kNNclf.fit(X_train, y_train)
        y_pred= kNNclf.predict(X_test)
        kNNaccr_scores[k]= metrics.accuracy_score(y_test, y_pred)
        print("Without cross validation accuracy is ", kNNaccr_scores[k] ," for K-value: ", k )
    
        cvscores = cross_val_score(kNNclf, X, y, cv=cvValue, scoring='accuracy')
        neglogcvscores = cross_val_score(kNNclf, X, y, cv=cvValue, scoring='neg_log_loss')
        rocauccvscores = cross_val_score(kNNclf, X, y, cv=cvValue, scoring='roc_auc')
        
        kNNcv_neglog_std.append(neglogcvscores.std())
        kNNcv_neglog_scores_mean.append(neglogcvscores.mean())
        
        kNNcv_rocauc_std.append(rocauccvscores.std())
        kNNcv_rocauc_scores_mean.append(rocauccvscores.mean())
        
        kNNcv_scores_std.append(cvscores.std())
        kNNcv_accr_scores_mean.append(cvscores.mean())
        print("With cross validation accuracy is ", kNNcv_accr_scores_mean[k-1])
        print("Log Loss is ", kNNcv_neglog_scores_mean[k-1])
        print("Area Under ROC Curve is ", kNNcv_rocauc_scores_mean[k-1])
    
    kNNaccr_scores = np.array(kNNaccr_scores)
    kNNcv_accr_scores_mean = np.array(kNNcv_accr_scores_mean)
    kNNcv_scores_std = np.array(kNNcv_scores_std)
    
    kNNcv_rocauc_scores_mean = np.array(kNNcv_rocauc_scores_mean)
    kNNcv_rocauc_std = np.array(kNNcv_rocauc_std)
    
    kNNcv_neglog_scores_mean = np.array(kNNcv_neglog_scores_mean)
    kNNcv_neglog_std = np.array(kNNcv_neglog_std)
            
    idx_max = kNNcv_accr_scores_mean.argmax()
    best_k_depth = k_range[idx_max]
    best_k_cv_score = kNNcv_accr_scores_mean[idx_max]
    best_k_cv_score_std = kNNcv_scores_std[idx_max]
    print('The k-{} kNN achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(best_k_depth, round(best_k_cv_score*100,5), round(best_k_cv_score_std*100, 5))) 
    
    idx_max = kNNcv_rocauc_scores_mean.argmax()
    best_k_depth = k_range[idx_max]
    best_k_cv_score = kNNcv_rocauc_scores_mean[idx_max]
    best_k_cv_score_std = kNNcv_rocauc_std[idx_max]
    print('The k-{} kNN achieves the best mean cross-validation roc auc {} +/- {}% on training dataset'.format(best_k_depth, round(best_k_cv_score*100,5), round(best_k_cv_score_std*100, 5))) 
    
    idx_max = kNNcv_neglog_scores_mean.argmax()
    best_k_depth = k_range[idx_max]
    best_k_cv_score = kNNcv_neglog_scores_mean[idx_max]
    best_k_cv_score_std = kNNcv_neglog_std[idx_max]
    print('The k-{} kNN achieves the best mean cross-validation log loss {} +/- {}% on training dataset'.format(best_k_depth, round(best_k_cv_score*100,5), round(best_k_cv_score_std*100, 5))) 
    
    
    kNNclf = KNeighborsClassifier(n_neighbors=best_k_depth, weights="uniform", algorithm="auto")
    kNNclf.fit(X_train, y_train)
    y_pred= kNNclf.predict(X_test)
    kNNaccr_scores[k]= metrics.accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    report = classification_report(y_test, y_pred)
    print(report)
    
    return kNNaccr_scores, kNNcv_accr_scores_mean, kNNcv_scores_std, kNNcv_neglog_scores_mean, kNNcv_neglog_std, kNNcv_rocauc_scores_mean, kNNcv_rocauc_std
    

def DecisionTreesClassifier(X, y, X_train, X_test, y_train, y_test, cvValue, tree_depths):
    print("------------DECISION TREE CLASSIFIER------------")
    treeaccr_scores=np.empty((40))
    treecv_accr_scores_mean = []
    treecv_scores_std = []#standart derivation
    treecv_neglog_std = []
    treecv_rocauc_std = []
    treecv_rocauc_scores_mean = []
    treecv_neglog_scores_mean = []
    for depth in tree_depths:
        treeClf = DecisionTreeClassifier(max_depth=depth)
        treeClf.fit(X_train, y_train)
        y_pred= treeClf.predict(X_test)
        treeaccr_scores[depth]= metrics.accuracy_score(y_test, y_pred)
        print("Without cross validation accuracy is ", treeaccr_scores[depth] ," for depth-value: ", depth )
        
        cvscores = cross_val_score(treeClf, X, y, cv=cvValue, scoring='accuracy')
        neglogcvscores = cross_val_score(treeClf, X, y, cv=cvValue, scoring='neg_log_loss')
        rocauccvscores = cross_val_score(treeClf, X, y, cv=cvValue, scoring='roc_auc')
        
        treecv_neglog_std.append(neglogcvscores.std())
        treecv_neglog_scores_mean.append(neglogcvscores.mean())
        
        treecv_rocauc_std.append(rocauccvscores.std())
        treecv_rocauc_scores_mean.append(rocauccvscores.mean())
         
        treecv_scores_std.append(cvscores.std())
        treecv_accr_scores_mean.append(cvscores.mean())
        print("With cross validation accuracy is ", treecv_accr_scores_mean[depth-1]," for depth: ", depth )
        print("Log Loss is ", treecv_neglog_scores_mean[depth-1])
        print("Area Under ROC Curve is ", treecv_rocauc_scores_mean[depth-1])

    treeaccr_scores = np.array(treeaccr_scores)
    treecv_accr_scores_mean = np.array(treecv_accr_scores_mean)
    treecv_scores_std = np.array(treecv_scores_std)
    
    treecv_rocauc_scores_mean = np.array(treecv_rocauc_scores_mean)
    treecv_rocauc_std = np.array(treecv_rocauc_std)
    
    treecv_neglog_scores_mean = np.array(treecv_neglog_scores_mean)
    treecv_neglog_std = np.array(treecv_neglog_std)
        
    idx_max = treecv_accr_scores_mean.argmax()
    best_tree_depth = tree_depths[idx_max]
    best_tree_cv_score = treecv_accr_scores_mean[idx_max]
    best_tree_cv_score_std = treecv_scores_std[idx_max]
    print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(best_tree_depth, round(best_tree_cv_score*100,5), round(best_tree_cv_score_std*100, 5)))
    
    idx_max = treecv_rocauc_scores_mean.argmax()
    best_tree_depth = tree_depths[idx_max]
    best_tree_cv_score = treecv_rocauc_scores_mean[idx_max]
    best_tree_cv_score_std = treecv_rocauc_std[idx_max]
    print('The depth-{} tree achieves the best mean cross-validation roc auc {} +/- {}% on training dataset'.format(best_tree_depth, round(best_tree_cv_score*100,5), round(best_tree_cv_score_std*100, 5)))
    
    idx_max = treecv_neglog_scores_mean.argmax()
    best_tree_depth = tree_depths[idx_max]
    best_tree_cv_score = treecv_neglog_scores_mean[idx_max]
    best_tree_cv_score_std = treecv_neglog_std[idx_max]
    print('The depth-{} tree achieves the best mean cross-validation log los {} +/- {}% on training dataset'.format(best_tree_depth, round(best_tree_cv_score*100,5), round(best_tree_cv_score_std*100, 5)))
    
    
    
    treeClf = DecisionTreeClassifier(max_depth=best_tree_depth)
    treeClf.fit(X_train, y_train)
    y_pred= treeClf.predict(X_test)
    treeaccr_scores[depth]= metrics.accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    report = classification_report(y_test, y_pred)
    print(report)
    tree.plot_tree(treeClf)#çizim
    
    return treeaccr_scores, treecv_accr_scores_mean, treecv_scores_std, treecv_neglog_scores_mean, treecv_neglog_std, treecv_rocauc_scores_mean, treecv_rocauc_std

def plot_cvScore(range,cv_accr_scores_mean, cv_scores_std, title, scoretype, model):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(range, cv_accr_scores_mean, '-o', label=scoretype, alpha=0.9)
    ax.fill_between(range, cv_accr_scores_mean-2*cv_scores_std, cv_accr_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(model, fontsize=14)
    ax.set_ylabel(scoretype, fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(range)
    ax.legend()

   
def main():
    (X, y) = data()
    model_range = range(1,25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
    kNNaccr_scores, kNNcv_accr_scores_mean, kNNcv_scores_std, kNNcv_neglog_scores_mean, kNNcv_neglog_std, kNNcv_rocauc_scores_mean, kNNcv_rocauc_std = kNNClassifier(X, y, X_train, X_test, y_train, y_test, 10, model_range)
    treeaccr_scores, treecv_accr_scores_mean, treecv_scores_std, treecv_neglog_scores_mean, treecv_neglog_std, treecv_rocauc_scores_mean, treecv_rocauc_std = DecisionTreesClassifier(X, y, X_train, X_test, y_train, y_test, 10, model_range)
    
    plot_cvScore(model_range, kNNcv_accr_scores_mean, kNNcv_scores_std, "Accuracy for k-Value on training data", 'Accuracy', "k-value")
    plot_cvScore(model_range, treecv_accr_scores_mean, treecv_scores_std, "Accuracy for decision tree depth on training data", 'Accuracy', "depth-value" )
    
    plot_cvScore(model_range, kNNcv_neglog_scores_mean, kNNcv_neglog_std, "Log Loss for k-Value on training data", 'Log Loss', "k-value")
    plot_cvScore(model_range, treecv_neglog_scores_mean, treecv_neglog_std, "Log Loss  for decision tree depth on training data", 'Log Loss', "depth-value")
    
    plot_cvScore(model_range, kNNcv_rocauc_scores_mean, kNNcv_rocauc_std, "Area Under ROC Curve for k-Value on training data", 'ROC AUC', "k-value")
    plot_cvScore(model_range, treecv_rocauc_scores_mean, treecv_rocauc_std, "Area Under ROC Curve  for decision tree depth on training data", 'ROC AUC', "depth-value")

    
if __name__ == "__main__":
    main()
