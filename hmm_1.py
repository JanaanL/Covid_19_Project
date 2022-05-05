import warnings
import itertools
import pandas as pd
import numpy as np
import math
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
 
 
# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
 
class CovidPredictor(object):
    def __init__(self, data_file, k_folds = 5, n_hidden_states=4, n_latency_days=10, pred_type = 'r'):
   
        self._data = pd.read_csv(data_file)
        self._k_folds = k_folds
        self._n_latency_days = n_latency_days
        self._pred_type = pred_type
        self.hmm = GaussianHMM(n_components=n_hidden_states)
        #self._possible_outcomes = np.arange(-1 * n_steps_change, n_steps_change)
        self._get_possible_outcomes()
        print("Creating Covid Predictor for metric {}".format(pred_type))


    def _get_possible_outcomes(self):
        
        if self._pred_type == 'r':
            column = np.array(self._data['Change Daily Cases'])
        elif self._pred_type == 'a':
            column = np.array(self._data['Change Active'])
        else:
            column = np.array(self._data['Change Deaths'])

        min_change = column.min()
        max_change = column.max()
        mean = column.mean()
        print("The minimum change is {} and the maximum change is {} and the mean is {}".format(min_change, max_change, mean))

        self._possible_outcomes = np.arange(min_change, max_change) 

    def _extract_features(self, data):

        if self._pred_type == 'r':
            change = np.array(data['Change Daily Cases'])
        elif self._pred_type == 'a':
            change = np.array(data['Change Active'])
        else:
            change= np.array(data['Change Deaths'])

        features = np.reshape(change ,(-1, 1))
        return features
 
    def _fit(self, train_data):
        feature_vector = self._extract_features(train_data)
        self.hmm.fit(feature_vector)

    def _measure_accuracy(self, predicted, actual):
        n = len(predicted)
        total_diffs = 0.0
        for i in range(n):
            total_diffs += abs(predicted[i] - actual[i])
        return total_diffs / n 
 
    def _predict_case_counts(self, test, day_index):
        previous_data_start_index = max(0, day_index - self._n_latency_days)
        previous_data_end_index = max(0, day_index)
        previous_data = test.iloc[previous_data_start_index: previous_data_end_index]
        previous_data_features = self._extract_features(previous_data)
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]
 
        return most_probable_outcome
        
    def _predict_cases_for_days(self, test, days):
        predicted_cases = []
        for day_index in range(days):
            predicted_cases.append(self._predict_case_counts(test, day_index))
        
        print("Predicted Cases: {}".format(predicted_cases))
        test_data = test[0:days]
        if self._pred_type == 'r':
            actual_cases = test_data['Change Daily Cases'].to_numpy()
        elif self._pred_type == 'a':
            actual_cases = test_data['Change Active'].to_numpy()
        else:
            actual_cases = test_data['Change Deaths'].to_numpy()
        #print("actual cases: {}".format(actual_cases))
        accuracy = self._measure_accuracy(predicted_cases, actual_cases)
        #print("The average total difference is {}".format(accuracy))

        return accuracy

    def cross_validation(self):
        #split train/test data
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
            self._fit(train)
            total_accuracy += self._predict_cases_for_days(test, 10)
        average_accuracy = total_accuracy / self._k_folds
        print("The average difference after {} k-fold validation is {}".format(self._k_folds, average_accuracy))

    def predict_without_cross_validation(self, test_size):
        train, test = train_test_split(self._data, test_size=test_size, shuffle=False)
        self._fit(train)
        accuracy = self._predict_cases_for_days(test, 10)
        print("The difference after training without cross validation is {}".format(accuracy))

for metric in ['r','a','d']:
    covid_predictor = CovidPredictor('covid_data.csv', pred_type = metric)
    #covid_predictor.predict_without_cross_validation(test_size = .20)
    covid_predictor.cross_validation()
    print("\n")
