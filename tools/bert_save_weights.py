from transformers import AutoModel
import numpy as np


def save_weights(model, file_name):
    state_dict = model.state_dict()
    weights = {}
    for k, v in state_dict.items():
        print(k)
        weights[k] = v
    np.savez(file_name, **weights)

if __name__=="__main__":
    model = AutoModel.from_pretrained("bert-base-uncased")
    save_weights(model, "BERT-B.npz")
    model = AutoModel.from_pretrained("bert-large-uncased")
    save_weights(model, "BERT-L.npz")
