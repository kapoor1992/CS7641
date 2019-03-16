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
from sklearn.decomposition import FactorAnalysis

def run_mamm():
    tc, X_train, X_test, y_train, y_test = get_mammography_data(100)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    fa = FactorAnalysis(random_state=0)  
    X_train = fa.fit_transform(X_train)  

    labels = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    y_pos = np.arange(1, 16)
    vars = fa.noise_variance_

    print(vars)

    plt.bar(y_pos, vars, align='center', alpha=0.5)
    plt.title('Mammography FA')
    plt.ylabel('Noise Variance')
    plt.xlabel('Factor Component')
    plt.xticks(np.arange(1, 16, 1))
    plt.show()

def run_skin():
    tc, X_train, X_test, y_train, y_test = get_skin_data(100)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)

    fa = FactorAnalysis(random_state=0)  
    X_train = fa.fit_transform(X_train)  

    labels = {'1', '2', '3'}
    y_pos = np.arange(1, 4)
    vars = fa.noise_variance_

    print(vars)

    plt.bar(y_pos, vars, align='center', alpha=0.5)
    plt.title('Skin FA')
    plt.ylabel('Noise Variance')
    plt.xlabel('Factor Component')
    plt.xticks(np.arange(1, 4, 1))
    plt.show()

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

run_mamm()
run_skin()