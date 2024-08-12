import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

from .model import Model

class DTreeModel(Model):
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

        self.x_train = self.learn_df.drop('default', axis=1)
        self.y_train = self.learn_df['default']

    def train(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.model = DecisionTreeClassifier()
        self.model.fit(self.x_train, self.y_train)

    def save_results(self):
        plt.figure()

        plot_tree(
            self.model,
            fontsize=8,
            max_depth=3,
            feature_names=['age','ed','employ','address','income','debtinc','creddebt','othdebt']
        )

        plt.show()