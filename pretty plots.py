import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

emotions_data = pd.read_csv('Podcast Data/labels_paths/train_labels.csv')
discrete_emotions = np.unique(emotions_data['EmoClass'].tolist())
indexes = []
for emot in discrete_emotions:
    if emot == 'A' or emot == 'U':
        continue
    if emot == 'C':
        indexes.append(['contempt', emotions_data[emotions_data.EmoClass == emot].first_valid_index()])
        continue
    indexes.append([emot, emotions_data[emotions_data.EmoClass == emot].first_valid_index()])
ax = plt.axes(projection='3d')

for emotion_label, row in indexes:
    ar = emotions_data.iloc[row]['EmoAct']
    val = emotions_data.iloc[row]['EmoVal']
    dom = emotions_data.iloc[row]['EmoDom']
    ax.scatter3D(ar, val, dom, cmap='viridis', label=emotion_label, s=40)

ax.legend(loc='lower left', bbox_to_anchor=(-0.35, 0.4), fontsize=10, title='Emotions')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
ax.set_xlim([1.0, 7.0])
ax.set_ylim([1.0, 7.0])
ax.set_zlim([1.0, 7.0])
plt.title('Emotions in the three-dimensional Russell and Mehrabian space')
# plt.show()
plt.savefig('emotions_in_3d.png')
