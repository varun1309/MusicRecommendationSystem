#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


# In[19]:


songs_triplet_df = pd.read_table("/Users/varun/Desktop/sml-project/MusicRecommendationSystem/10000.txt", header=None)
songs_triplet_df.columns = ['user_id', 'song_id', 'listen_count']
songs_metadata_df =  pd.read_csv("/Users/varun/Desktop/sml-project/MusicRecommendationSystem/song_data.csv")


# In[20]:


songs_triplet_df.head()


# In[21]:


songs = pd.merge(songs_triplet_df, songs_metadata_df.drop_duplicates(['song_id']), on="song_id", how="left")
songs.head()


# In[25]:


songs = songs.head(10000)


# In[32]:


users = songs['user_id'].unique()
unique_all_songs = songs['song_id'].unique()


# In[29]:



train_data, test_data = train_test_split(songs, test_size = 0.15, random_state=0)


# In[52]:


def get_user_songs(user_id, training_data):
    user_songs = training_data[training_data["user_id"] == user_id]
    user_unique_songs = list(user_songs["song_id"].unique())
    return user_unique_songs


# In[53]:


def get_song_user(song_id, training_data):
    song_user = training_data[training_data['song_id'] == song_id]
    song_unique_users  = list(song_user["user_id"].unique())
    return song_unique_users


# In[54]:


print(get_song_user(unique_all_songs[0], train_data))


# In[84]:


def create_cooccurence_matrix(user_id, training_data, unique_all_songs, user_songs):
#     user_songs = get_user_songs(user_id, training_data)
    song_users = []
    for i in range(0, len(user_songs)):
        song_users.append(get_song_user(user_songs[i], training_data))
    
    song_co_occurence_mat = np.matrix(np.zeros(shape=(len(user_songs), len(unique_all_songs))), float)
#     print (song_co_occurence_mat.shape)
    
    for i in range(0, len(unique_all_songs)):
        current_song_users = set(get_song_user(unique_all_songs[i], training_data))
#         print('i = ', i)
        for j in range(0, len(user_songs)):
#             print('j= ', j)
            j_song_users = set(song_users[j])
            common_users = current_song_users.intersection(j_song_users)
            if len(common_users) > 0:
#                 print (song_co_occurence_mat)
                all_users = current_song_users.union(j_song_users)
                score = float(len(common_users))/float(len(all_users))
                song_co_occurence_mat[j,i] = score
            else:
                song_co_occurence_mat[j,i] = 0
    return song_co_occurence_mat


# In[100]:


def recommender(user_id, training_data, unique_all_songs):
    user_songs = get_user_songs(user_id, training_data)
    song_co_occurence_mat = create_cooccurence_matrix(user_id, training_data, unique_all_songs, user_songs)
    scores = song_co_occurence_mat.sum(axis = 0) / float(song_co_occurence_mat.shape[0])
    scores = np.array(scores)[0].tolist()
    
    song_sorted_indices = sorted(((v,i) for i,v in enumerate(list(scores))), reverse=True)
    output_cols = ['user_id', 'song_id', 'score']
    recommended_df = pd.DataFrame(columns=output_cols)
    score = 1
    
    for i in range(0,len(song_sorted_indices)):
        if ~np.isnan(song_sorted_indices[i][0]) and unique_all_songs[song_sorted_indices[i][1]] not in user_songs and score <= 10:
            recommended_df.loc[len(recommended_df)]=[user_id,unique_all_songs[song_sorted_indices[i][1]],song_sorted_indices[i][0]]
            score += 1
    
    return recommended_df
    


# In[101]:



recommended_df = recommender(users[0], train_data, unique_all_songs)
recommended_df= pd.merge(recommended_df, songs_metadata_df, on='song_id', how='left')
recommended_df


# In[125]:


def evaluate(train_data, test_data):
    train_users = set(train_data['user_id'].unique())
    test_users = set(test_data['user_id'].unique())
    common_users = train_users.intersection(test_users)
    print(len(common_users))
    
    N = 1
    from collections import defaultdict
    recommendation_dict = defaultdict(int)
    for i in range(0, N):
        count = 0
        for user in common_users:
            recommended_songs = set(recommender(user, train_data, unique_all_songs)['song_id'])
            test_songs = set(test_data[test_data["user_id"] == user]['song_id'])
#             print(recommended_df, test_songs)
            train_test_hit_count = len(test_songs.intersection(recommended_songs))
            recommendation_dict[train_test_hit_count] = recommendation_dict[train_test_hit_count] + 1
            print(count)
            count += 1
    
    for k, v in recommendation_dict.items():
        print(k, " - ", v)
    
    


# In[126]:


evaluate(train_data, test_data)


# In[ ]:




