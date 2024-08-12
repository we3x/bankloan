from models.SVModel import SVModel
from models.DTreeModel import DTreeModel
from models.LogRegModel import LogRegModel
from models.DTreeModel import DTreeModel


# Model podpornih vektora
BankloanSVM = SVModel(
    learn_path='data/bankloan-learn.csv',
    test_path='data/bankloan-test.csv'
)

# Model logisticke regresije
BankloanLR = LogRegModel(
    learn_path='data/bankloan-learn.csv',
    test_path='data/bankloan-test.csv'
)

BankloanDT = DTreeModel(
    learn_path='data/bankloan-learn.csv',
    test_path='data/bankloan-test.csv'
)

for model in [BankloanSVM, BankloanLR, BankloanDT]:

    model.load_data()

    model.split_data()

    model.train()

    model.test_model()

    model.predict()

    model.save_results()
