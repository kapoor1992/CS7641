import numpy
import sklearn
import matplotlib
import pandas

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# MAMMOGRAPHY
# 162 missing values
# 130 lines removed
# ages transformed from years to decades

# HAPPINESS
# moved label from first to last in columns

#data = pandas.read_csv('/Users/admin/Downloads/happiness_cleaned.csv', ',', encoding = 'utf-16')
data = pandas.read_csv('/Users/admin/Downloads/mammography_cleaned.csv', ',', encoding = 'utf-8')
attribute_cols = ['BI_RADS', 'age', 'shape', 'margin', 'density']
attributes = data.loc[:, attribute_cols]
labels = data.label

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(attributes, labels)
