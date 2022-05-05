import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
 
 
# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
 
 
class CovidPredictor(object):
    def __init__(self, train_data, test_data, order=1, metric='r'):

        self._train = train_data
        self._test = test_data
        self._order = order
        self._metric = metric
        self._label_encoder()

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
        if isinstance (data['Combined Labels'], str):
            features = self._le.transform(np.array([data['Combined Labels']]).reshape(-1,1)).astype('int')
        else:
            features = self._le.transform(np.array(data['Combined Labels']).reshape(-1,1)).astype('int')

        return features.flatten()
 
    def learn_model(self):
        feature_vector = self._extract_features(self._train)
        
        epsilon = 0.001
        n_labels = len(self._le.categories_[0])

        #1st order Markov chain
        if self._order == 1:
            probs = np.empty((n_labels, n_labels))
            probs.fill(epsilon)
            priorLabel = None

            for i,label in enumerate(feature_vector):
                if i == 0:
                    priorLabel = feature_vector[0]
                else:
                    currentLabel = feature_vector[i]
                    probs[priorLabel][currentLabel] +=1
                    priorLabel = currentLabel
            
            probs = probs / probs.sum(axis=1, keepdims=True)

        #2nd order Markov chain
        else:
            probs = np.empty((n_labels, n_labels, n_labels))
            probs.fill(epsilon)
            priorLabels = []

            for i, label in enumerate(feature_vector):
                if i < 2:
                    priorLabels.append(label)
                else:
                    currentLabel = feature_vector[i]
                    probs[priorLabels[0]][priorLabels[1]][currentLabel] += 1
                    priorLabels = [priorLabels[1], currentLabel]

            probs = probs / probs.sum(axis=2, keepdims=True)

        self._probs = probs

    def _get_most_probable_outcome(self, day_index):

        if self._order == 1:
            previous_data = self._test.iloc[day_index - 1]
            previous_label = self._extract_features(previous_data)
            predicted_label = np.random.choice(self._probs.shape[0],p=self._probs[previous_label][0])

        else:
            previous_data = self._test.iloc[day_index-2:day_index]
            previous_label = self._extract_features(previous_data)
            predicted_label = np.random.choice(self._probs.shape[0], p=self._probs[previous_label[0]][previous_label[1]])

        return predicted_label
                 
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
 

    def predict(self, days):
        predicted_codes = []
        if self._order == 1:
            n_prior_days = 1
        else:
            n_prior_days = 2
        for day_index in range(n_prior_days, days + n_prior_days):
            predicted_code = self._get_most_probable_outcome(day_index)
            predicted_codes.append(predicted_code)
        predicted_labels = self._le.inverse_transform(np.array(predicted_codes).reshape(-1,1)).flatten()
        
        #print("Predicted Labels: {}".format(predicted_cases, predicted_changes))
        test_data = self._test[n_prior_days:days+n_prior_days]
        if self._metric == 'r':
            actual_labels = test_data['Daily Cases Label'] + test_data['Change Daily Cases Label']
        elif self._metric == 'a':
            actual_labels = test_data['Estimated Active Label'] + test_data['Change Active Label']
        else:
            actual_labels = test_data['Current Deaths Label'] + test_data['Change Deaths Label']
    
        accuracy = self._measure_accuracy(predicted_labels, np.array(actual_labels))
        print("The total accuracy is {}".format(accuracy))
        return accuracy

    def get_baseline(self, days):
        print("Calculating baseline score for metric type {}".format(self._metric))
        n_labels = len(self._le.categories_[0])
        #calculate probabilities
        epsilon = 0.001
        probs = np.empty(n_labels)
        probs.fill(epsilon)
        feature_vector = self._extract_features(self._train)
        for label in feature_vector:
            probs[label] +=1
        probs = probs / probs.sum()

        #Make predictions based on probabilities
        predictions = []
        for i in range(days):
            predictions.append(np.random.choice(n_labels, p=probs))
        predicted_labels = self._le.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()
        
        test_data = self._test[0:days]
        actual_labels = test_data['Daily Cases Label'] + test_data['Change Daily Cases Label']
        accuracy = self._measure_accuracy(predicted_labels, np.array(actual_labels))
        print("The total accuracy is {}".format(accuracy))

def predict_without_cross_validation(data_file, test_size, order=1, metric='r'):
    data = pd.read_csv(data_file)
    train, test = train_test_split(data, test_size=test_size, shuffle=False)
    covid_predictor = CovidPredictor(train, test, order=order,  metric=metric)
    covid_predictor.get_baseline(10)
    covid_predictor.learn_model()
    accuracy = covid_predictor.predict(10)
    print("The accuracy after training without cross validation is {}".format(accuracy))


def cross_validation(data_file, k_folds=4, order=1, metric='r'):
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
        covid_predictor = CovidPredictor(train, test, order=order, metric=metric)
        covid_predictor.learn_model()
        total_accuracy += covid_predictor.predict(10)
    average_accuracy = total_accuracy / (k_folds - 1)
    print("The average accuracy after {} k-fold validation is {}".format(k_folds, average_accuracy))


predict_without_cross_validation('covid_data.csv', test_size = .20, order=1, metric = 'r')
cross_validation('covid_data.csv', k_folds = 5, order=2, metric = 'r')
 


