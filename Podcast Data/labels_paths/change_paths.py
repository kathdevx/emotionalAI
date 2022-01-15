for i in range(0, 5):
    fname = 'paths_to_train_wavs_batch' + str(i) + '.txt'
    with open(fname, 'r') as infile:
        lines = infile.read().splitlines()
        new_lines = []
        for line in lines:
            new = line.replace('/content/drive/Shareddrives/DNN -  audio files/Podcasts_Dataset/Data/Audio Files/',
                               'Audios/')
            new_lines.append(new)

        with open('new_' + fname, 'w') as f:
            for item in new_lines:
                f.write("%s\n" % item)
