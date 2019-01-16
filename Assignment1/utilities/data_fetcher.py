import pandas

# 162 missing values
# 130 lines removed
# ages transformed from years to decades
def get_mammography_data(train_percent):
    data = pandas.read_csv('utilities/data/mammography_data.csv', ',', encoding = 'utf-8')

    attribute_cols = ['BI_RADS', 'age', 'shape', 'margin', 'density']
    label_col = ['diagnosis']
    train_count = (int)(train_percent * 830 / 100)

    train_attributes = data.loc[:train_count - 1, attribute_cols]
    train_labels = data.loc[:train_count - 1, label_col]

    test_attributes = data.loc[train_count:, attribute_cols]
    test_labels = data.loc[train_count:, label_col]

    return train_count, train_attributes, train_labels, test_attributes, test_labels

# moved label from first to last in columns
def get_happiness_data(train_percent):
    data = pandas.read_csv('utilities/data/happiness_data.csv', ',', encoding = 'utf-16')

    attribute_cols = ['city_info','housing_cost','school_quality','police_trust','street_maintenance','community_events']
    label_col = ['happy']
    train_count = (int)(train_percent * 143 / 100)

    train_attributes = data.loc[:train_count - 1, attribute_cols]
    train_labels = data.loc[:train_count - 1, label_col]

    test_attributes = data.loc[train_count:, attribute_cols]
    test_labels = data.loc[train_count:, label_col]

    return train_count, train_attributes, train_labels, test_attributes, test_labels