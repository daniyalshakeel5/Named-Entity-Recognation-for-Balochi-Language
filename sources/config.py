import simpletransformers
from simpletransformers.ner import NERArgs
from sklearn.metrics import accuracy_score
import pandas as pd

args = NERArgs()
args.num_train_epochs = 15  # how many generations
args.learning_rate = 1e-4  # at what speed the model learns
args.overwrite_output_dir = True
args.train_batch_size = 32  # size of the training batch
args.eval_batch_size = 32  # size if the evaluation batch after training
acc_score = accuracy_score
pickled_file = "uncased_model.pkl"

file = pd.read_csv('C:/Users/Lenovo/Downloads/person name.csv')
