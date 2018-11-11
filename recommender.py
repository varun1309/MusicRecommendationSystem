import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# print(songs_summary.head()[['title', 'song_id']])

song_df_1 = pd.read_table("10000.txt", header=None)

# song_df_2 = pd.read_csv(songs_metadata_file)

print(song_df_1.head())

n_u = len(song_df_1[0].unique())
n_m = len(song_df_1[1].unique())

sparsity = len(song_df_1)/(n_u*n_m)

train_data, test_data = train_test_split(song_df_1, test_size=0.2)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

train_data[2] = (train_data[2] - train_data[2].mean()) / train_data[2].std()


vectorizer = CountVectorizer()
vectorizer.fit_transform(song_df_1[0].unique())
user_dict = vectorizer.vocabulary_

vectorizer1 = CountVectorizer()
vectorizer1.fit_transform(song_df_1[1].unique())
song_dict = vectorizer1.vocabulary_



# Creating training matrix
training_matrix = np.zeros((n_u, n_m))
for line in train_data.itertuples():
    training_matrix[user_dict[line[1].lower()], song_dict[line[2].lower()]] = line[3]

# Creating test matrix
testing_matrix = np.zeros((n_u, n_m))
for line in test_data.itertuples():
    testing_matrix[user_dict[line[1].lower()], song_dict[line[2].lower()]] = line[3]

print(training_matrix[0])
