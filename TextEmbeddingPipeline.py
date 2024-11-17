import torch
from transformers import pipeline
from datetime import datetime
import os
import json
from datasets import load_from_disk, load_dataset
import numpy as np
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import textwrap
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

class EmbedFlow:
    def __init__(self, model_name, dataset_name="IMDB", prefix="Prefix: ", suffix=" :Suffix"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.models = {
            "BERT": "bert-base-uncased",
            "ST": "sentence-transformers/all-MiniLM-L12-v2",
            "T5": "t5-base",
            "INS": "hkunlp/instructor-large"
        }
        self.datasets = {
            "IMDB": "imdb"
        }
        self.train_data = None
        self.test_data = None
        self.model_name = model_name
        self.dataset_name = dataset_name.lower()
        self.prefix = prefix
        self.suffix = suffix
        self.truncation = True
        self.padding = True
        self.max_length = 512
        self.dataset_key = "text"

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not available. Choose from {list(self.models.keys())}.")
        if dataset_name.upper() not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(self.datasets.keys())}.")
        
        self.model = pipeline("feature-extraction", model=self.models[model_name], device=self.device)

    def load_sample_data(self, sample_size=5000):
        dataset = load_dataset(self.datasets[self.dataset_name.upper()])]
        if(sample_size!=0):
            self.train_data = dataset['train'].shuffle(seed=42).select(range(sample_size))
            self.test_data = dataset['test'].shuffle(seed=42).select(range(sample_size))

        self.train_data.save_to_disk('sampled_train_data')
        self.test_data.save_to_disk('sampled_test_data')
        print("Data loaded and sampled.")

    def load_local_data(self):
        try:
            self.train_data = load_from_disk('sampled_train_data')
            self.test_data = load_from_disk('sampled_test_data')
            print("Local data loaded.")
        except Exception as e:
            print(f"Error loading local data: {e}")

    def augment_data(self):
        self.train_data = self.train_data.map(self.modify_example)
        #self.test_data = self.test_data.map(self.modify_example)
        print("Data augmented with prefixes and suffixes.")

    def modify_example(self, example):
        example['text'] = self.prefix + example['text'] + self.suffix
        return example

    def embed_data(self, data, use_mean_pooling=False):
        data_key = KeyDataset(data, self.dataset_key)
        pipe = self.model(data_key, return_tensors=True, truncation=self.truncation, padding=self.padding, max_length=self.max_length)
        embeddings = []
        for tensor in tqdm(pipe, desc="Embedding text"): 
            if use_mean_pooling:
                tensor = tensor.mean(dim=1).flatten()
            embeddings.append(tensor.numpy())
        return np.array(embeddings), np.array(data["label"])

    def save_data(self, embeddings, labels, save_path="data"):
        timestamp = datetime.now().strftime("%m-%d_%H:%M")
        avg_shape = np.mean([tensor.shape for tensor in embeddings], axis=0).tolist()
    
        embedding_info = {
            'model_name': self.model_name,
            'num_embeddings': len(embeddings),
            'avg_embedding_shape': avg_shape,
            'created_at': timestamp
        }
    
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"{self.model_name}_embeddings.npy"), embeddings)
        np.save(os.path.join(save_path, f"{self.model_name}_labels.npy"), labels)
        with open(os.path.join(save_path, f"{self.model_name}_metadata.json"), 'w') as f:
            json.dump(embedding_info, f)
    
        print(f"Embeddings and labels saved at {timestamp}.")

    def load_data(self, save_path="data"):
        embeddings = np.load(os.path.join(save_path, f"{self.model_name}_embeddings.npy"))
        labels = np.load(os.path.join(save_path, f"{self.model_name}_labels.npy"))
        with open(os.path.join(save_path, f"{self.model_name}_metadata.json"), 'r') as f:
            metadata = json.load(f)
    
        print(f"Data loaded for model: {metadata['model_name']}")
        return embeddings, labels

    def evaluate(self, method='svm', save_path="data"):
        embeddings, labels = self.load_data(save_path)
        train_size = int(0.8 * len(embeddings))
        train_embeddings, train_labels = embeddings[:train_size], labels[:train_size]
        test_embeddings, test_labels = embeddings[train_size:], labels[train_size:]

        if method == 'svm':
            model = SVC(kernel='linear')
        elif method == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                                  solver='sgd', verbose=1, random_state=1,
                                  learning_rate_init=.1)
        else:
            raise ValueError("Choose either 'svm' or 'mlp' for evaluation.")

        model.fit(train_embeddings, train_labels)
        predictions = model.predict(test_embeddings)
        print(f"{method.upper()} Evaluation Report:")
        print(classification_report(test_labels, predictions))

    def start_flow(self, sample_size=5000, use_mean_pooling=False, method='svm', save_path="data"):
        self.load_sample_data(sample_size)
        self.load_local_data()
        self.augment_data()
        train_embeddings, train_labels = self.embed_data(self.train_data, use_mean_pooling=use_mean_pooling)
        self.save_data(train_embeddings, train_labels, save_path)
        self.evaluate(method, save_path)

# Example usage:
def main():
    embedder = EmbedFlow(model_name="BERT", dataset_name="IMDB", prefix="Prefix: ", suffix=" :Suffix")
    embedder.start_flow(sample_size=1000, use_mean_pooling=True, method='svm')

if __name__ == "__main__":
    main()