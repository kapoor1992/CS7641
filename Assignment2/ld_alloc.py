from sklearn.decomposition import LatentDirichletAllocation
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

def run_mamm():
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_mammography_data(100)
    df = pd.DataFrame(train_attributes)
    scaler=StandardScaler()
    scaler.fit(df)
    scaled_data=scaler.transform(df)

    X = scale(train_attributes)
    y = pd.DataFrame(train_labels)

    lda=LatentDirichletAllocation(n_components=2, random_state=0)
    lda.fit(df)
    x_lda=lda.transform(df)

    color_theme = np.array(['magenta', 'brown'])
    plt.title('Mammography Latent Dirichlet Allocation')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.scatter(x_lda[:,0],x_lda[:,1], c=color_theme[train_labels.values[:,0]],s=5)
    plt.show()

def run_skin():
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_skin_data(1)
    df = pd.DataFrame(train_attributes)
    scaler=StandardScaler()
    scaler.fit(df)
    scaled_data=scaler.transform(df)

    X = scale(train_attributes)
    y = pd.DataFrame(train_labels)

    lda=LatentDirichletAllocation(n_components=2, random_state=0)
    lda.fit(df)
    x_lda=lda.transform(df)

    color_theme = np.array(['magenta', 'brown'])
    plt.title('Skin Latent Dirichlet Allocation')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.scatter(x_lda[:,0],x_lda[:,1], c=color_theme[train_labels.values[:,0]],s=5)
    plt.show()

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

run_mamm()
run_skin()