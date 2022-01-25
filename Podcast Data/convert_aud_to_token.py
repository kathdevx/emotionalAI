import os

import torch

import argparse

import soundfile as sf

from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel

# problem_aud = open('vq-wav2vec-Kmeans-Roberta/PROBLEM_AUD1.text', 'w')


class EmotionDataPreprocessing:

    def __init__(self):

        cp = torch.load('vq-wav2vec-Kmeans-Roberta/vq-wav2vec_kmeans.pt')
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()

        # Roberta wav2vec
        self.roberta = RobertaModel.from_pretrained('vq-wav2vec-Kmeans-Roberta', checkpoint_file='bert_kmeans.pt')

        self.roberta.eval()

    def indices_to_string(self, idxs):
        # based on fairseq/examples/wav2vec/vq-wav2vec_featurize.py
        return "<s>" + " " + " ".join("-".join(map(str, a.tolist())) for a in idxs.squeeze(0))

    def preprocess_audio_file(self, filename):

        wav, curr_sample_rate = sf.read(filename)

        feats_audio = torch.from_numpy(wav).float()

        assert feats_audio.dim() == 1, feats_audio.dim()
        # print("Audio: ",feats_audio.size())
        return feats_audio

    def preprocess_data(self, audio_path):
        num_items = 1e18
        current_num = 0

        # AUDIO
        if audio_path:
            # all_audio_features = []
            # audio_files = sorted(glob.glob(audio_path+"*.wav"))
            with open(audio_path, 'r') as f:
                audio_files = f.read().splitlines()
            print(len(audio_files), " audio_files found")

            for audio_file in audio_files:
                print(f'{current_num + 1}/{len(audio_files)} done')
                output_file = audio_file.replace('Audio', 'Audio Token').replace('.pt', '.txt')
                if os.path.exists(output_file) or output_file == 'Audio Tokens/MSP-PODCAST_1023_0235.wav':
                    # print(f'{output_file} already exists!')
                    current_num += 1
                    if current_num > num_items:
                        break
                    continue
                print(f'{current_num}/{len(audio_files) - 1} was not in there!')

                audio_features = self.preprocess_audio_file(audio_file).unsqueeze(0)

                # wav2vec
                z = self.model.feature_extractor(audio_features)

                _, idxs = self.model.vector_quantizer.forward_idx(z)

                idx_str = self.indices_to_string(idxs)

                tokens = self.roberta.task.source_dictionary.encode_line(idx_str, append_eos=True,
                                                                         add_if_not_exist=False).cpu().detach().numpy()

                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    for item in tokens:
                        f.write(str(item) + '\t')
                current_num += 1
                if current_num > num_items:
                    break


if __name__ == "__main__":
    data_processor = EmotionDataPreprocessing()

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--audio_path', default=None, help='path for raw audio files')

    args = parser.parse_args()

    audio_path = args.audio_path

    data_processor.preprocess_data(audio_path)
