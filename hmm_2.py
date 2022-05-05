import warnings
import itertools
import pandas as pd
import numpy as np
import math
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import MultinomialHMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
 
 
# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
 
class CovidPredictor(object):
    def __init__(self, train_data, test_data, n_hidden_states=4, n_latency_days=10, metric='r'):

        self._test = test_data
        self._train = train_data
        self._n_latency_days = n_latency_days
        self._metric = metric
        self.hmm = MultinomialHMM(n_components=n_hidden_states)
        self._label_encoder()
        self._possible_outcomes = self._le.transform(self._le.categories_[0].reshape(-1,1)).astype('int')

    def _label_encoder(self):
        #Extract labels and concatenate them
        data = self._train
        if self._metric == 'r':
            data['Combined Labels'] = data['Daily Cases Label'] + data['Change Daily Cases Label']
        elif self._metric == 'a':
            data['Combined Labels'] = data['Estimated Active Label'] + data['Change Active Label']
        else:
            data['Combined Labels'] = data['Current Deaths Label'] + data['Change Deaths Label']
        
        self._le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        features =  np.array(data['Combined Labels'])
        self._le.fit(features.reshape(-1,1))
        
    def _extract_features(self, data):
        if self._metric == 'r':
            data['Combined Labels'] = data['Daily Cases Label'] + data['Change Daily Cases Label']
        elif self._metric == 'a':
            data['Combined Labels'] = data['Estimated Active Label'] + data['Change Active Label']
        else:
            data['Combined Labels'] = data['Current Deaths Label'] + data['Change Deaths Label']
        features = self._le.transform(np.array(data['Combined Labels']).reshape(-1,1)).astype('int')
        return features.reshape(-1,1)
 
    def fit(self):
        feature_vector = self._extract_features(self._train)
        self.hmm.fit(feature_vector)

    def _measure_accuracy(self, predicted, actual):
        n = len(predicted)
        label_map = {'L': 1, 'M': 2, 'H':3, 'E':4}
        total_accuracy = 0.0
        for i in range(n):
            p_label = predicted[i]
            a_label = actual[i]
            #calculate score for Daily Label:
            diff_daily_label = abs(label_map.get(predicted[i][0]) - label_map.get(actual[i][0]))
            accuracy_1 = (1 - diff_daily_label / 4) * 2
            
            #calcaulate score for change label
            diff_change_label = abs(label_map[predicted[i][1]] - label_map[actual[i][1]])
            accuracy_2 = 1 - diff_change_label / 3 
            
            #calculate score for +/-
            if predicted[i][2] == actual[i][2]:
                accuracy_3 = 1
            else:
                accuracy_3 = 0.5

            total_accuracy += (accuracy_1 + accuracy_2 + accuracy_3) / 4
        return total_accuracy / n

    def predict_case_counts(self, day_index):
        previous_data_start_index = max(0, day_index - self._n_latency_days)
        previous_data_end_index = max(1, day_index)
        previous_data = self._test.iloc[previous_data_start_index: previous_data_end_index]
        previous_data_features = self._extract_features(previous_data)
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = np.argmax(outcome_score)
 
        return most_probable_outcome

    def predict(self, days):
        predicted_codes = []
        for day_index in range(days):
            predicted_code = self.predict_case_counts(day_index)
            predicted_codes.append(predicted_code)
        predicted_labels = self._le.inverse_transform(np.array(predicted_codes).reshape(-1,1)).flatten()
        test_data = self._test[0:days]
        if self._metric == 'r':
            actual_labels = test_data['Daily Cases Label'] + test_data['Change Daily Cases Label']
        elif self._metric == 'a':
            actual_labels = test_data['Estimated Active Label'] + test_data['Change Active Label']
        else:
            actual_labels = test_data['Current Deaths Label'] + test_data['Change Deaths Label']
#        print("Predicted labels: ", predicted_labels)
#        print("Actual labels: ", actual_labels)
#        print("Predicted codes: ", predicted_codes)
#        print("Actual codes: ", self._le.transform(np.array(actual_labels).reshape(-1,1)))
    
        accuracy = self._measure_accuracy(predicted_labels, np.array(actual_labels))
        print("The total accuracy is {}".format(accuracy))

        return accuracy
 


def predict_without_cross_validation(data_file, test_size, metric='r'):
    data = pd.read_csv(data_file)
    train, test = train_test_split(data, test_size=test_size, shuffle=False)
    covid_predictor = CovidPredictor(train, test, metric=metric)
    covid_predictor.fit()
    accuracy = covid_predictor.predict(20)
    print("The accuracy after training without cross validation is {}".format(accuracy))


def cross_validation(data_file, k_folds=4, metric='r'):
    data = pd.read_csv(data_file)
    size = len(data)
    k_fold_size = size // k_folds
    train_data = []
    test_data = []
    for i in range(k_folds - 1):
        index = k_fold_size * (i + 1)
        train_data.append(data[:index])
        test_data.append(data[index:k_fold_size * (i+2)])

    total_accuracy = 0
    for train, test in zip(train_data, test_data):
        covid_predictor = CovidPredictor(train, test, metric=metric)
        covid_predictor.fit()
        total_accuracy += covid_predictor.predict(10)
    average_accuracy = total_accuracy / (k_folds - 1)
    print("The average accuracy after {} k-fold validation is {}".format(k_folds, average_accuracy))

#predict_without_cross_validation('covid_data.csv', test_size = .20, metric = 'a')
cross_validation('covid_data.csv', k_folds = 4, metric = 'd')
