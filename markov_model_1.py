import warnings
import itertools
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
 
# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
 
class CovidPredictor(object):
    def __init__(self, data_file, k_folds = 5, order=1, metric='r'):

        self._data = pd.read_csv(data_file)
        self._k_folds = k_folds
        self._order = order
        self._metric = metric
        self._get_bias_and_size()
        print("Creating Covid Predictor for metric {}".format(metric))

    def _get_bias_and_size(self):
        
        if self._metric == 'r':
            column = np.array(self._data['Change Daily Cases'])
        elif self._metric == 'a':
            column = np.array(self._data['Change Active'])
        else:
            column = np.array(self._data['Change Deaths'])

        min_change = column.min()
        max_change = column.max()
        self._bias = -1 * min_change
        self._size = abs(max_change - min_change)
    
    def _extract_features(self, data):
        if self._metric == 'r':
            change = np.array(data['Change Daily Cases'])
        elif self._metric == 'a':
            change = np.array(data['Change Active'])
        else:
            change= np.array(data['Change Deaths'])

        return change.flatten()
 
    def _learn_model(self, train_data):
        feature_vector = self._extract_features(train_data)
        epsilon = 0.001

        #1st order Markov chain
        if self._order == 1:
            probs = np.empty((self._size, self._size))
            probs.fill(epsilon)
            priorChange = None

            for i in range(len(feature_vector)):
                if i == 0:
                    priorChange = feature_vector[0]
                else:
                    currentChange = feature_vector[i]
                    probs[priorChange + self._bias][currentChange + self._bias] +=1
                    priorChange = currentChange
            
            probs = probs / probs.sum(axis=1, keepdims=True)

        #2nd order Markov chain
        else:
            probs = np.empty((self._size, self._size, self._size))
            probs.fill(epsilon)
            priorChanges = []

            for i, change in enumerate(feature_vector):
                if i < 2:
                    priorChanges.append(change)
                else:
                    currentChange = change
                    probs[priorChangess[0]+bias][priorChangess[1] + self._bias][currentChange + self._bias] += 1
                    priorChanges = [priorChanges[1], currentChange]

            probs = probs / probs.sum(axis=2, keepdims=True)

        self._probs = probs

    def _get_most_probable_outcome(self, test_data, day_index):

        if self._order == 1:
            previous_data = test_data.iloc[day_index - 1]
            previous_change = self._extract_features(previous_data)
            predicted_change = np.random.choice(self._probs.shape[0],p=self._probs[previous_change + self._bias][0])

        else:
            previous_data = self._test_data.iloc[day_index-2:day_index]
            previous_change = self._extract_features(previous_data)
            predicted_change = np.random.choice(self._probs.shape[0], p=self._probs[previous_change[0] + self._bias][previous_change[1] + self._bias])

        return predicted_change - self._bias
                 
    def _measure_accuracy(self, predicted, actual):
        n = len(predicted)
        total_diffs = 0.0
        for i in range(n):
            total_diffs += abs(predicted[i] - actual[i])
        return total_diffs / n

    def _predict_changes_for_days(self, test_data, days):
        predicted_changes = []
        if self._order == 1:
            n_prior_days = 1
        else:
            n_prior_days = 2
        for day_index in range(n_prior_days, days + n_prior_days):
            predicted_change = self._get_most_probable_outcome(test_data, day_index)
            predicted_changes.append(predicted_change)
        
        test_data = test_data[n_prior_days:days+n_prior_days]
        if self._metric == 'r':
            actual_changes = test_data['Change Daily Cases'].to_numpy()
        elif self._metric == 'a':
            actual_changes = test_data['Change Active'].to_numpy()
        else:
            actual_changes = test_data['Change Deaths'].to_numpy()

        accuracy = self._measure_accuracy(predicted_changes, np.array(actual_changes))
        return accuracy

    def get_baseline(self, days):
        
        size = len(self._data)
        k_fold_size = size // self._k_folds
        train_data = []
        test_data = []
        for i in range(self._k_folds - 1):
            index = k_fold_size * (i + 1)
            train_data.append(self._data[:index])
            test_data.append(self._data[index:k_fold_size * (i+2)])

        total_accuracy = 0
        for train, test in zip(train_data, test_data):
        
            #calculate probabilities
            epsilon = 0.001
            probs = np.empty(self._size)
            probs.fill(epsilon)
            feature_vector = self._extract_features(train)
            for change in feature_vector:
                probs[change] +=1
            probs = probs / probs.sum()

            #Make predictions based on probabilities
            predictions = []
            for i in range(days):
                predictions.append(np.random.choice(self._size, p=probs))
            
            test_data = test[0:days]
            if self._metric == 'r':
                actual_changes = test_data['Change Daily Cases'].to_numpy()
            elif self._metric == 'a':
                actual_changes = test_data['Change Active'].to_numpy()
            else:
                actual_changes = test_data['Change Deaths'].to_numpy()
            total_accuracy += self._measure_accuracy(predictions, np.array(actual_changes))
        
        average_accuracy = total_accuracy / self._k_folds
        print("The baseline score is {:2f}".format(average_accuracy))
  
    def cross_validation(self, n_days = 40):
        size = len(self._data)
        k_fold_size = size // self._k_folds
        train_data = []
        test_data = []
        for i in range(self._k_folds - 1):
            index = k_fold_size * (i + 1)
            train_data.append(self._data[:index])
            test_data.append(self._data[index:k_fold_size * (i+2)])

        total_accuracy = 0
        for train, test in zip(train_data, test_data):
            self._learn_model(train)
            total_accuracy += self._predict_changes_for_days(test, n_days)
        average_accuracy = total_accuracy / self._k_folds
        print("The average difference after {} k-fold validation is {}".format(self._k_folds, average_accuracy))

for metric in ['r','a','d']:
    covid_predictor = CovidPredictor('covid_data.csv', order=1, metric=metric)
    covid_predictor.get_baseline(10)
    covid_predictor.cross_validation(10)
    print("\n")
