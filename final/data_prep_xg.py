import h5py
import numpy as np

import pickle


#data creation from already created files for xg_boost
#requires user_dict, song_dict, normalized files for traning created in matrix_factorization.py
user_dict=np.load('user_dict.npy')
user_list=user_dict.tolist()
inv_map_user = {v: k for k, v in user_list.items()}
song_dict=np.load('song_dict.npy')

song_list=song_dict.tolist()
inv_map_song = {v: k for k, v in song_list.items()}
h5f = h5py.File('training_freq_matrix.h5','r')
array_train = h5f['name-of-dataset-training'][:]
h5f.close()

i=0
data=[]
for row in array_train:
    non_zero_index=[j for j, e in enumerate(row) if e != 0]

    user=inv_map_user[i]
    for song_index in non_zero_index:

        song=inv_map_song[song_index]
        rating=row[song_index]
        data.append([user,song,rating])



    i+=1
    pass

with open('outfile_for_xg', 'wb') as fp:
    pickle.dump(data, fp)
pass
