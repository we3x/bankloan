import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from .model import Model

class SVModel(Model):
    def __init__(
        self,
        learn_path: str,
        test_path: str,
    ):
        self.scaler = StandardScaler()
        self.model = None
        self.learn_path = learn_path
        self.test_path = test_path

    def load_data(self):
        self.learn_df = pd.read_csv(self.learn_path)
        self.test_df = pd.read_csv(self.test_path)

    def split_data(self, test_size=0.1, random_state=28):
        x_axis = self.learn_df.drop('default', axis=1)
        y_axis = self.learn_df['default']

        x_train, x_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=test_size, random_state=random_state)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self, kernel="linear"):
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)

        self.model = SVC(kernel=kernel)  
        self.model.fit(self.x_train_scaled, self.y_train)

    def test_model(self):
        y_pred = self.model.predict(self.x_test_scaled)
        acc_score = accuracy_score(self.y_test, y_pred)

        print(f'\n\nSVM Tacnost: {acc_score}')
        print(f'Izvestaj o klasifikaciji:\n{classification_report(self.y_test, y_pred)}')

        correct_predictions_index = []
        for row_index, (input, prediction, label) in enumerate(zip (self.x_test_scaled, y_pred, self.y_test)):
            if prediction == label:
                correct_predictions_index.append(row_index)

        wrong_predictions = self.x_test.drop(index=[self.x_test.index[index] for index in correct_predictions_index])

        wrong_predictions['should_predicted'] = self.y_test.drop(index=[self.y_test.index[index] for index in correct_predictions_index])

        print(wrong_predictions)

        return acc_score

    def predict(self):
        test_df_scaled = self.scaler.transform(self.test_df)
        self.predictions = self.model.predict(test_df_scaled)

    def save_results(self, result_path:str = 'results/results-svm.csv'):
        predicted_data = self.test_df
        predicted_data['default'] = self.predictions

        predicted_data.to_csv(result_path, sep=",", index=False)
