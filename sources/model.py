from simpletransformers.ner import NERModel
import pandas as pd
import dataset
import config
import pickle
import torch
import numpy

train_data, test_data, label = dataset.data_set(config.file, "NER")


def model_training():
    model = NERModel('bert', 'bert-base-uncased', labels=label, args=config.args, use_cuda=False)
    model.train_model(train_data, eval_data=test_data, acc=config.acc_score)
    result, model_outputs, pred_lists = model.eval_model(test_data)

    filename = config.pickled_file
    pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
    model_training()
