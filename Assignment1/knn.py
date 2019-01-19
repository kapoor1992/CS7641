from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from utilities.data_fetcher import *

def train(train_attributes, train_labels, neighbours, kernel):
    classifier = KNeighborsClassifier(n_neighbors = neighbours, weights = kernel)
    classifier.fit(train_attributes, train_labels.values.ravel())

    return classifier

def run(train_attributes, train_labels, test_attributes, test_labels):
    neighbours_list = [1, 3, 5, 7]
    kernels = ['uniform', 'distance']
    folds = 4

    print('\n')

    for neighbours in neighbours_list:
        for kernel in kernels:
            print('neighbours: ' + str(neighbours))
            print('kernel: ' + kernel)

            classifier = train(train_attributes, train_labels, neighbours, kernel)

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

print('-- HAPPINESS --')
for train_percentage in train_percentages:
        print('train percent: ' + str(train_percentage))
        train_count, train_attributes, train_labels, test_attributes, test_labels = get_happiness_data(train_percentage)

        print('train count: ' + str(train_count))
        run(train_attributes, train_labels, test_attributes, test_labels)