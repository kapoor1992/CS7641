from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from utilities.data_fetcher import *

def train(train_attributes, train_labels, impurity_threshold):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, min_impurity_decrease = impurity_threshold)
    classifier.fit(train_attributes, train_labels)

    return classifier

def run(train_attributes, train_labels, test_attributes, test_labels):
    impurity_thresholds = [0, 0.1, 0.2, 0.3, 0.4]
    folds = 4

    print('\n')

    for impurity_threshold in impurity_thresholds:
        print('impurity threshold: ' + str(impurity_threshold))

        classifier = train(train_attributes, train_labels, impurity_threshold)

        cross_valid_score = cross_val_score(classifier, train_attributes, train_labels, cv = folds)
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