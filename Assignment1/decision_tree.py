from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from utilities.data_fetcher import *

def train(train_attributes, train_labels, impurity_threshold):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, min_impurity_decrease = impurity_threshold)
    classifier.fit(train_attributes, train_labels)

    return classifier

def run(train_attributes, train_labels, test_attributes, test_labels):
    impurity_thresholds = [0, 0.0001, 0.001, 0.01, 0.1]
    folds = 4

    print('\n')

    for impurity_threshold in impurity_thresholds:
        print('impurity threshold: ' + str(impurity_threshold))

        classifier = train(train_attributes, train_labels, impurity_threshold)
        print('node count: ' + str(classifier.tree_.node_count))

        cross_valid_score = cross_val_score(classifier, train_attributes, train_labels, cv = folds)
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