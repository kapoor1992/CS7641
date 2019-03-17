from utilities.data_fetcher import *
from sklearn.preprocessing import scale
from sklearn.exceptions import DataConversionWarning
from warnings import filterwarnings
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_mamm():
    tc, X_train, X_test, y_train, y_test = get_mammography_data(100)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    pca = PCA()  
    X_train = pca.fit_transform(X_train)  

    labels = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    y_pos = np.arange(1, 16)
    vars = pca.explained_variance_ratio_

    print(vars)

    plt.bar(y_pos, vars, align='center', alpha=0.5)
    plt.title('Mammography PCA')
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Component')
    plt.xticks(np.arange(1, 16, 1))
    plt.show()

def run_skin():
    tc, X_train, X_test, y_train, y_test = get_skin_data(100)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    pca = PCA()  
    X_train = pca.fit_transform(X_train)  

    labels = {'1', '2', '3'}
    y_pos = np.arange(1, 4)
    vars = pca.explained_variance_ratio_

    print(vars)

    plt.bar(y_pos, vars, align='center', alpha=0.5)
    plt.title('Skin PCA')
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Component')
    plt.xticks(np.arange(1, 4, 1))
    plt.show()

def get_mamm(train_percentage=100):
    tc, X_train, X_test, y_train, y_test = get_mammography_data(train_percentage)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    
    pca = PCA(n_components=9)  
    X_train = pca.fit_transform(X_train)

    pca2 = PCA(n_components=9)  
    y_train = pca2.fit_transform(y_train)

    return tc, X_train, X_test, y_train, y_test

def get_skin():
    tc, X_train, X_test, y_train, y_test = get_skin_data(10)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    pca = PCA(n_components=1)  
    X_train = pca.fit_transform(X_train)

    return tc, X_train, X_test, y_train, y_test

warnings.filterwarnings(action='ignore', category=DataConversionWarning)