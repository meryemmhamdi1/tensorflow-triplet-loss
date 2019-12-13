import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def process_bert_tokenize(text):
    marked_text = "[CLS] " + text + " [SEP]"

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    print("Loading Model")

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

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

    return token_embeddings


if __name__ == "__main__":
    text = "Here is the sentence I want embeddings for."
    print(process_bert_tokenize(text))
