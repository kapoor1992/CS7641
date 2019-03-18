import pandas

# 162 missing values
# 130 lines removed
# ages transformed from years to decades and nominal attributes made binary
def get_mammography_data(train_percent):
    data = pandas.read_csv('utilities/data/mammography_data.csv', ',', encoding = 'utf-8')

    attribute_cols = ['BI_RADS','age','shape (round)','shape (oval)','shape (lobular)','shape (irregular)','margin (circumscribed)','margin (microlobulated)','margin (obscured)','margin (ill defined)','margin (spiculated)','density (high)','density (iso)','density (low)','density (fat containing)']
    label_col = ['diagnosis']
    train_count = (int)(train_percent * 830 / 100)

    train_attributes = data.loc[:train_count - 1, attribute_cols]
    train_labels = data.loc[:train_count - 1, label_col]

    test_attributes = data.loc[train_count:, attribute_cols]
    test_labels = data.loc[train_count:, label_col]

    return train_count, train_attributes, train_labels, test_attributes, test_labels

# shuffled rows
def get_skin_data(train_percent):
    data = pandas.read_csv('utilities/data/skin_data.csv', ',', encoding = 'utf-8')

    attribute_cols = ['blue','green','red']
    label_col = ['is_non_skin']
    train_count = (int)(train_percent * 245057 / 100)

    train_attributes = data.loc[:train_count - 1, attribute_cols]
    train_labels = data.loc[:train_count - 1, label_col]

    test_attributes = data.loc[train_count:, attribute_cols]
    test_labels = data.loc[train_count:, label_col]

    return train_count, train_attributes, train_labels, test_attributes, test_labels

def get_mamm_clust_reduced(train_percent, path):
    data = pandas.read_csv(path, ',', encoding = 'utf-8')

    attribute_cols = ['a','b','c','d','e','f','g']
    label_col = ['h']
    train_count = (int)(train_percent * 830 / 100)

    train_attributes = data.loc[:train_count - 1, attribute_cols]
    train_labels = data.loc[:train_count - 1, label_col]

    test_attributes = data.loc[train_count:, attribute_cols]
    test_labels = data.loc[train_count:, label_col]

    return train_count, train_attributes, train_labels, test_attributes, test_labels

def get_kmeans_pca(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/kmeans_pca.csv')

def get_kmeans_ica(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/kmeans_ica.csv')

def get_kmeans_rp(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/kmeans_rp.csv')

def get_kmeans_fa(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/kmeans_fa.csv')

def get_exmax_pca(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/exmax_pca.csv')

def get_exmax_ica(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/exmax_ica.csv')

def get_exmax_rp(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/exmax_rp.csv')

def get_exmax_fa(train_percent):
    return get_mamm_clust_reduced(train_percent, 'utilities/data/exmax_fa.csv')