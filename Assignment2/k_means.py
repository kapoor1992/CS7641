from utilities.data_fetcher import *
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.exceptions import DataConversionWarning
from warnings import filterwarnings
import warnings

def run_mamm():
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_mammography_data(100)
    X = scale(train_attributes)
    y = pd.DataFrame(train_labels)

    clustering = KMeans(n_clusters=2, random_state=0)
    clustering.fit(X)

    mamm_df = pd.DataFrame(train_attributes)
    mamm_df.columns = ['BI_RADS','age','shape (round)','shape (oval)','shape (lobular)','shape (irregular)','margin (circumscribed)','margin (microlobulated)','margin (obscured)','margin (ill defined)','margin (spiculated)','density (high)','density (iso)','density (low)','density (fat containing)']
    y.columns == ['diagnosis']

    color_theme = np.array(['magenta', 'brown'])
    plt.scatter(x=mamm_df.age, y=mamm_df.BI_RADS, c=color_theme[train_labels.values[:,0]], s=5)
    plt.show()

    plt.scatter(x=mamm_df.age, y=mamm_df.BI_RADS, c=color_theme[clustering.labels_], s=5)
    plt.show()

def run_skin():
    train_count, train_attributes, train_labels, test_attributes, test_labels = get_skin_data(100)
    X = scale(train_attributes)
    y = pd.DataFrame(train_labels)

    clustering = KMeans(n_clusters=2, random_state=0)
    clustering.fit(X)

    skin_df = pd.DataFrame(train_attributes)
    skin_df.columns = ['blue','green','red']
    y.columns == ['is_non_skin']
    
    color_theme = np.array(['purple', 'brown'])
    
    plt.scatter(x=skin_df.green, y=skin_df.red, c=color_theme[train_labels.values[:,0]], s=5)
    plt.show()
    
    plt.scatter(x=skin_df.green, y=skin_df.red, c=color_theme[clustering.labels_], s=5)
    plt.show()

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

run_mamm()
run_skin()