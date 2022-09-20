"""Basic example: getting word embeddings from CharacterBERT"""
from transformers import BertTokenizer
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer

# Example text
x = "Hello World!"

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained(
    './pretrained-models/bert-base-uncased/')
x = tokenizer.basic_tokenizer.tokenize(x)

# Add [CLS] and [SEP]
x = ['[CLS]', *x, '[SEP]']

# Convert token sequence into character indices
indexer = CharacterIndexer()
batch = [x]  # This is a batch with a single token sequence x
batch_ids = indexer.as_padded_tensor(batch)

# Load some pre-trained CharacterBERT
model = CharacterBertModel.from_pretrained(
    './pretrained-models/bert-base-uncased/')

# Feed batch to CharacterBERT & get the embeddings
embeddings_for_batch, _ = model(batch_ids)
embeddings_for_x = embeddings_for_batch[0]
print('These are the embeddings produces by CharacterBERT (last transformer layer)')
for token, embedding in zip(x, embeddings_for_x):
    print(token, embedding)
