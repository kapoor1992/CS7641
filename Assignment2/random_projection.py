from sklearn.random_projection import GaussianRandomProjection
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
import re

def run_mamm():
    eps_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dimens = []

    for i in eps_vals:
        tc, X_train, X_test, y_train, y_test = get_mammography_data(100)
        grp = GaussianRandomProjection(random_state=0)
        grp.set_params(eps=i)
        sc = StandardScaler()

        try:
            X_train = grp.fit(X_train)
        except ValueError as err: 
            val = re.search('of(.*)which', str(err))
            dimens.append(int(val.group(1)))

    print(dimens)

    plt.title('Mammography Random Projection')
    plt.xlabel('eps')
    plt.ylabel('Required Dimensions')
    plt.scatter(x=eps_vals,y=dimens)
    plt.show()
    

def run_skin():
    eps_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dimens = []

    for i in eps_vals:
        tc, X_train, X_test, y_train, y_test = get_skin_data(100)
        grp = GaussianRandomProjection(random_state=0)
        grp.set_params(eps=i)
        sc = StandardScaler()

        try:
            X_train = grp.fit(X_train)
        except ValueError as err: 
            val = re.search('of(.*)which', str(err))
            dimens.append(int(val.group(1)))

    print(dimens)

    plt.title('Skin Random Projection')
    plt.xlabel('eps')
    plt.ylabel('Required Dimensions')
    plt.scatter(x=eps_vals,y=dimens)
    plt.show()

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def get_mamm(train_percentage=100):
    tc, X_train, X_test, y_train, y_test = get_mammography_data(train_percentage)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)
    try:
        y_train = sc.fit_transform(y_train)
    except:
        pass

    grp = GaussianRandomProjection(n_components=14, random_state=0)  
    X_train = grp.fit_transform(X_train)

    try:
        grp2 = GaussianRandomProjection(n_components=14, random_state=0)  
        y_train = grp2.fit_transform(y_train)
    except:
        pass

    return tc, X_train, X_test, y_train, y_test

def get_skin():
    tc, X_train, X_test, y_train, y_test = get_skin_data(10)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    grp = GaussianRandomProjection(n_components=2, random_state=0)  
    X_train = grp.fit_transform(X_train)

    return tc, X_train, X_test, y_train, y_test
