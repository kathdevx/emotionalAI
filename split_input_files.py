import pandas as pd

labels = pd.read_csv('Podcast Data/labels_paths/labels_concensus.csv')
split_sets = labels['Split_Set'].unique()

for split_set in split_sets:
    split_df = labels[labels['Split_Set'] == split_set]
    fname = 'Podcast Data/labels_paths/' + split_set + '.csv'
    split_df.to_csv(fname)
