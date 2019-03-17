from sklearn.decomposition import FastICA
from utilities.data_fetcher import *
from sklearn.preprocessing import scale
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from warnings import filterwarnings
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_mamm():
    scores = []

    for i in range(1, 16):
        tc, X_train, X_test, y_train, y_test = get_mammography_data(100)
        ica = FastICA(random_state=0)
        ica.set_params(n_components=i)
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)
        X_train = ica.fit_transform(X_train)
        frame = pd.DataFrame(X_train)
        frame = frame.kurt(axis=0)
        scores.append(frame.abs().mean())
        print(frame.abs().mean())
    
    plt.bar(np.arange(1, 16), scores, align='center', alpha=0.5)
    plt.title('Mammography ICA')
    plt.ylabel('Kurtosis')
    plt.xlabel('Independent Component Count')
    plt.xticks(np.arange(1, 16, 1))
    plt.show()

def run_skin():
    scores = []

    for i in range(1, 4):
        tc, X_train, X_test, y_train, y_test = get_mammography_data(100)
        ica = FastICA(random_state=0)
        ica.set_params(n_components=i)
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)
        X_train = ica.fit_transform(X_train)
        frame = pd.DataFrame(X_train)
        frame = frame.kurt(axis=0)
        scores.append(frame.abs().mean())
        print(frame.abs().mean())
    
    plt.bar(np.arange(1, 4), scores, align='center', alpha=0.5)
    plt.title('Skin ICA')
    plt.ylabel('Kurtosis')
    plt.xlabel('Independent Component Count')
    plt.xticks(np.arange(1, 4, 1))
    plt.show()

def get_mamm():
    tc, X_train, X_test, y_train, y_test = get_mammography_data(100)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    ica = FastICA(n_components=9, random_state=0)  
    X_train = ica.fit_transform(X_train)

    return X_train

def get_skin():
    tc, X_train, X_test, y_train, y_test = get_skin_data(10)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    ica = FastICA(n_components=2, random_state=0)  
    X_train = ica.fit_transform(X_train)

    return X_train

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)