import os
import pickle

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

for i in range(0, 230):
    train_file = 'Podcast Data/training sentences/train_sentences_chunk_' + str(i) + '.txt'
    output_file = 'Podcast Data/Sentence Tokens/tokenized_train_sentences_chunk_' + str(i) + '.pkl'
    encoded_sentences = []
    if os.path.exists(output_file):
        print(f'{i + 1}/230 done!')
        continue
    with open(train_file, 'r') as infile:
        training_sentences = infile.read().splitlines()
        for training_sentence in training_sentences:
            corresponding_file, sentence = training_sentence.split(' ', 1)
            encoded_input = tokenizer(sentence, return_tensors='pt')
            encoded_sentences.append([corresponding_file, encoded_input])
        with open(output_file, 'wb') as outfile:
            pickle.dump(encoded_sentences, outfile)
    print(f'{i + 1}/230 done!')
