import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def data_set(df):
    data = df
    data = data.fillna(method="ffill")
    data['Sentence #'] = LabelEncoder().fit_transform(data["Sentence #"])
    data.rename(columns={'NER': 'labels'}, inplace=True)
    X = data[["Sentence #", "Words"]]
    Y = data['labels']

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)

    train_data = pd.DataFrame({"sentence_id": xtrain["Sentence #"], "words": xtrain["Words"], "labels": ytrain})
    test_data = pd.DataFrame({"sentence_id": xtest["Sentence #"], "words": xtest["Words"], "labels": ytest})

    labels = data['labels'].unique().tolist()

    return train_data, test_data, labels

# def pos_data_set(df):
#     data = df
#     data = data.fillna(method="ffill")
#     data['Sentence_id'] = LabelEncoder().fit_transform(data["Sentence_id"])
#     data.rename(columns={'POS': 'labels'})

#     X = data[["Sentence #", "Words"]]
#     Y = data['labels']

#     xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)

#     train_data = pd.DataFrame({"sentence_id": xtrain["Sentence #"], "words": xtrain["Words"], "labels": ytrain})
#     tes_data = pd.DataFrame({"sentence_id": xtest["Sentence #"], "words": xtest["Words"], "labels": ytest})

#     labels = data['labels'].unique().tolist()

#     return train_data, tes_data, labels
