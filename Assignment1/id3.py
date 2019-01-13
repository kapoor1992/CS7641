import numpy
import sklearn
import matplotlib
import pandas
from utilities.data_fetcher import get_mammography_data
from utilities.data_fetcher import get_happiness_data

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

train_attributes, train_labels, test_attributes, test_labels = get_happiness_data()

print(train_labels.shape)
print(test_labels.shape)

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(train_attributes, train_labels)
