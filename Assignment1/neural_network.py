from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from utilities.data_fetcher import *

def train(train_attributes, train_labels):
    classifier = MLPClassifier(activation = 'logistic', solver = 'sgd', random_state = 0, max_iter = 10000, learning_rate_init = 0.001, alpha = 0.5)
    classifier.fit(train_attributes, train_labels.values.ravel())

    return classifier

def run(train_attributes, train_labels, test_attributes, test_labels):
    folds = 4

    print('\n')

    classifier = train(train_attributes, train_labels)

    cross_valid_score = cross_val_score(classifier, train_attributes, train_labels.values.ravel(), cv = folds)
    print('cross val scores: ' + str(cross_valid_score))
    print('avg cross val score: ' + str(sum(cross_valid_score) / folds))

    predictions = classifier.predict(test_attributes)
    
    test_score = classifier.score(test_attributes, test_labels)
    print('test score: ' + str(test_score))

    print('\n')

print('-- MAMMOGRAPHY --')
train_attributes, train_labels, test_attributes, test_labels = get_mammography_data()
run(train_attributes, train_labels, test_attributes, test_labels)

print('-- HAPPINESS --')
train_attributes, train_labels, test_attributes, test_labels = get_happiness_data()
run(train_attributes, train_labels, test_attributes, test_labels)