import pandas as pd
import warnings
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import glob
import torch
import soundfile as sf
import librosa
import os
import nltk
nltk.download('punkt')


def load_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        'facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
    return tokenizer, model


def correct_sentence(input_text):
    sentences = nltk.sent_tokenize(input_text)
    return (' '.join([s.replace(s[0], s[0].capitalize(), 1) for s in sentences]))


def asr_transcript(input_file):
    tokenizer, model = load_model()

    speech, fs = sf.read(input_file)

    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]

    if fs != 16000:
        speech = librosa.resample(speech, fs, 16000)

    input_values = tokenizer(speech, return_tensors="pt").input_values
    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = tokenizer.decode(predicted_ids[0])

    return correct_sentence(transcription.lower())


def chunks(lst, n):
    chunky = []
    for i in range(0, len(lst), n):
        chunky.append(lst[i:i + n])
    return chunky


test_split_sets = [pd.read_csv('Podcast Data/labels_paths/Test1.csv'),
                   pd.read_csv('Podcast Data/labels_paths/Test2.csv')]

batch_size = 1000

batches = [chunks(range(0, len(test_split_sets[0]) + 1), batch_size),
           chunks(range(0, len(test_split_sets[1]) + 1), batch_size)]
test1_files = glob.glob('Podcast Data/test sentences/test1_txt_chunk_*.txt')
test2_files = glob.glob('Podcast Data/test sentences/test2_txt_chunk_*.txt')


for test_num, _ in enumerate(test_split_sets):

    try:

        if len(test1_files) < len(batches[0]):
            latest_file = max(test1_files, key=os.path.getctime)

            test_num = int(latest_file.split('/')[-1].split('_')[0][-1])

            chunk = int(latest_file.split('_')[-1].split('.')[0])

            print(f'Load Checkpoint for test1 chunk {chunk}')

        elif len(test2_files) < len(batches[1]):
            latest_file = max(test2_files, key=os.path.getctime)

            test_num = int(latest_file.split('/')[-1].split('_')[0][-1])

            chunk = int(latest_file.split('_')[-1].split('.')[0])

            print(f'Load Checkpoint for test2 chunk {chunk}')

    except ValueError:
        if len(test1_files) < len(batches[0]):
            n = str(1)
            test_num = 1
        elif len(test2_files) < len(batches[1]):
            n = str(2)
            test_num = 2

        latest_file = None
        print(f'Did not find checkpoint. Starting from the top..')
        chunk = 0
        pass

    # chunk_num = input('Chunk num: ')
    # chunk_num = int(chunk_num)

    for chunk_num in range(chunk, len(batches[test_num-1])):

        if latest_file is None:
            latest_file = 'Podcast Data/test sentences/test' + n + '_txt_chunk_0.txt'
            it = 0
        else:

            latest_file = 'Podcast Data/test sentences/test' + \
                str(test_num) + '_txt_chunk_'+str(chunk_num)+'.txt'
            if os.path.exists(latest_file):
                print(f'{latest_file} exists')
                with open(latest_file, 'r') as infile:
                    lines = infile.read().splitlines()
                    if it > batches[test_num-1][chunk_num].stop:
                        continue
                    elif it > 0:
                        it -= 1
                        print(
                            f'Resuming from element: {it+1}/{batches[test_num-1][chunk_num].stop-batches[test_num-1][chunk_num].start}')
                with open(latest_file, 'w') as outfile:
                    if it > 0:
                        for line in lines[:-1]:
                            in_str = line + '\n'
                            outfile.write(in_str)
            else:
                # print(f'{latest_file} doesnt exist')
                it = 0

        batch_frame = test_split_sets[test_num-1].iloc[batches[test_num-1][chunk_num]
                                                       .start:batches[test_num-1][chunk_num].stop, :]

        with open(latest_file, 'a') as outfile:
            for name, path in zip(batch_frame.loc[it:, 'FileName'], batch_frame.loc[it:, 'Path']):
                finput = f'{name} {asr_transcript(path)} \n'
                outfile.write(finput)

                print(
                    f'Chunk {chunk_num+1}/{len(batches[test_num-1])-1} - {it+1}/{len(batch_frame)} done')
                it += 1

    print(f'Test {test_num} set done!')
