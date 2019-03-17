from warnings import filterwarnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from utilities.data_fetcher import *
import matplotlib.pyplot as plt
import numpy as np
import pca, ica, random_projection, factor_an

def train_orig(train_attributes, train_labels, iterations):
    classifier = MLPClassifier(activation = 'logistic', solver = 'sgd', random_state = 0, max_iter = iterations)
    classifier.fit(train_attributes, train_labels.values.ravel())

    return classifier

def train(train_attributes, train_labels, iterations):
    classifier = MLPClassifier(activation = 'logistic', solver = 'sgd', random_state = 0, max_iter = iterations)
    train_attributes = train_attributes.astype('int')
    train_labels = train_labels.astype('int')
    classifier.fit(train_attributes, train_labels)

    return classifier

def run(train_attributes, train_labels, test_attributes, test_labels, tp, orig=False):
    iterations_list = [1, 5, 10, 15]
    train_scores = []
    test_scores = []

    print('\n')

    for iterations in iterations_list:
        print('iterations: ' + str(iterations))
        
        if orig:
            classifier = train_orig(train_attributes, train_labels, iterations)
        else:
            classifier = train(train_attributes, train_labels, iterations)

        train_score = classifier.score(train_attributes, train_labels)
        print('training score: ' + str(train_score))
        train_scores.append(train_score)

        predictions = classifier.predict(test_attributes)
        
        test_score = classifier.score(test_attributes, test_labels)
        print('test score: ' + str(test_score))
        test_scores.append(test_score)

        print('\n')

    plt.title('Mammography Neural Network: Training Percentage = ' + str(tp) + '%')
    plt.xticks([1, 5, 10, 15])
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.plot(iterations_list, train_scores, label='Training Set')
    plt.plot(iterations_list, test_scores, label='Testing Set')
    plt.legend()
    plt.show()

def run_mamm_orig(train_percentages):
    for train_percentage in train_percentages:
            train_count, train_attributes, train_labels, test_attributes, test_labels = get_mammography_data(train_percentage)
            run(train_attributes, train_labels, test_attributes, test_labels, train_percentage, orig=True)

filterwarnings("ignore", category=ConvergenceWarning)
