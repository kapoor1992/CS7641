from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from utilities.data_fetcher import *
from warnings import filterwarnings

def train(train_attributes, train_labels, iterations, in_kernel):
    classifier = SVC(random_state = 0, gamma = 'auto', degree = 2, max_iter = iterations, kernel = in_kernel)
    classifier.fit(train_attributes, train_labels.values.ravel())

    return classifier

def run(train_attributes, train_labels, test_attributes, test_labels):
    iterations_list = [1, 5, 10, 15]
    kernels = ['linear', 'poly']
    folds = 4

    print('\n')

    for iterations in iterations_list:
        for kernel in kernels:
                print('iterations: ' + str(iterations))
                print('kernel: ' + kernel)

                classifier = train(train_attributes, train_labels, iterations, kernel)

                cross_valid_score = cross_val_score(classifier, train_attributes, train_labels.values.ravel(), cv = folds)
                print('cross val scores: ' + str(cross_valid_score))
                print('avg cross val score: ' + str(sum(cross_valid_score) / folds))

                predictions = classifier.predict(test_attributes)
                
                test_score = classifier.score(test_attributes, test_labels)
                print('test score: ' + str(test_score))

                print('\n')

filterwarnings("ignore", category=ConvergenceWarning)

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