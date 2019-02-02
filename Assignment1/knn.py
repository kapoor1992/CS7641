from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from utilities.data_fetcher import *

def train(train_attributes, train_labels, neighbours):
    classifier = KNeighborsClassifier(n_neighbors = neighbours)
    classifier.fit(train_attributes, train_labels.values.ravel())

    return classifier

def run(train_attributes, train_labels, test_attributes, test_labels):
    neighbours_list = [1, 2, 3, 4]
    folds = 4

    print('\n')

    for neighbours in neighbours_list:
        print('neighbours: ' + str(neighbours))

        classifier = train(train_attributes, train_labels, neighbours)
        print('training score: ' + str(classifier.score(train_attributes, train_labels)))
        
        cross_valid_score = cross_val_score(classifier, train_attributes, train_labels.values.ravel(), cv = folds)
        print('cross val scores: ' + str(cross_valid_score))
        print('avg cross val score: ' + str(sum(cross_valid_score) / folds))

        predictions = classifier.predict(test_attributes)
        
        test_score = classifier.score(test_attributes, test_labels)
        print('test score: ' + str(test_score))

        print('\n')

train_percentages = [20, 40, 60, 80]

print('-- MAMMOGRAPHY --')
for train_percentage in train_percentages:
        print('train percent: ' + str(train_percentage))
        train_count, train_attributes, train_labels, test_attributes, test_labels = get_mammography_data(train_percentage)

        print('train count: ' + str(train_count))
        run(train_attributes, train_labels, test_attributes, test_labels)

print('-- SKIN --')
for train_percentage in train_percentages:
        print('train percent: ' + str(train_percentage))
        train_count, train_attributes, train_labels, test_attributes, test_labels = get_skin_data(train_percentage)

        print('train count: ' + str(train_count))
        run(train_attributes, train_labels, test_attributes, test_labels)