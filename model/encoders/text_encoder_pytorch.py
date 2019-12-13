from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch

import pandas as pd
from numpy import load
from numpy import savez_compressed
import numpy as np
from tqdm import tqdm

BERT_MODEL = 'bert-base-multilingual-uncased'


def process_bert_tokenize(text):

    marked_text = "[CLS] " + text + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    print("Evaluating the model")
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)

    print("token_embeddings:", token_embeddings)
    print("token_embeddings.shape:", token_embeddings.shape)

    return token_embeddings.numpy()


if __name__ == "__main__":
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    # Load pre-trained model (weights)
    print("Loading Model")
    model = BertModel.from_pretrained(BERT_MODEL)

    # Load dataset
    print("Loading data")
    data = pd.read_csv("tweets_images_user_id_triplets.csv")

    print("Getting embeddings for the tweets")
    tweet_arr_vec = []
    for text in tqdm(list(data["tweets"])):
        tweet_arr_vec.append(process_bert_tokenize(text))

    tweet_arr_np = np.array(tweet_arr_vec)

    print("Saving the embeddings in a numpy array")

    savez_compressed('/Users/d22admin/USCGDrive/BeyondAssignment/Deliverables/Embeddings/text_embeddings.npz',
                     tweet_arr_vec)


