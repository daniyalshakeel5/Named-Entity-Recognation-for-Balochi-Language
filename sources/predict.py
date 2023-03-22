import pickle
import torch



def get_predictions(input):

    with open ('Trained_Model.pkl', 'rb') as f:
       model = pickle.load(f)
    string = input
    
    prediction, model_output = model.predict([string])

    return prediction