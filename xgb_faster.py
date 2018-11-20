import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cross_validation, grid_search, metrics, ensemble
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle

songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

with open ('outfile_for_xg', 'rb') as fp:
    itemlist = pickle.load(fp)

#print(itemlist)
user_id = []
song_id = []
percentile = []
for i in range(len(itemlist)):
    user_id.append(itemlist[i][0])
    song_id.append(itemlist[i][1].upper())
    percentile.append(itemlist[i][2])

values = np.rint(percentile)
#print(len(user_id))
#print(len(song_id))
#print(len(values))

song_df_1 = pd.DataFrame()

user = pd.Series(user_id)
song = pd.Series(song_id)
percentile = pd.Series(values)

song_df_1['user_id'] = user.values
song_df_1['song_id'] = song.values
song_df_1['listen_count'] = percentile.values
song_df_1=song_df_1[0:100]
song_df_2 =  pd.read_csv('song_data.csv')
#print(song_df_1)
#print(song_df_2)

song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
song_df = song_df.drop(["title", "release"],1)
#print(song_df)
print(song_df.columns)

for col in song_df.select_dtypes(include=['object']).columns:
    print("Converting")
    song_df[col] = song_df[col].astype('category')
    

for col in song_df.select_dtypes(include=['category']).columns:
    print("Converting")
    song_df[col] = song_df[col].cat.codes

target = song_df.pop('listen_count')
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(song_df, target, test_size = 0.3)
del song_df

model = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=250, verbose_eval=True)
model.fit(train_data, train_labels)

pickle.dump(model, open("xg.pickle.dat", "wb"))
loaded_model = pickle.load(open("xg.pickle.dat", "rb"))
print(test_data)
predict_labels = loaded_model.predict(test_data)
print(predict_labels)
accuracy = accuracy_score(test_labels, predict_labels)
print(accuracy)

