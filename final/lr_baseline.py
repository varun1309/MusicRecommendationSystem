import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cross_validation, grid_search, metrics, ensemble
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_auc_score

#Import the song dataset
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

#Read the normalized values file
with open ('outfile_for_xg', 'rb') as fp:
    itemlist = pickle.load(fp)

#Fetch the normalized song counts
user_id = []
song_id = []
percentile = []
for i in range(len(itemlist)):
    user_id.append(itemlist[i][0])
    song_id.append(itemlist[i][1].upper())
    percentile.append(itemlist[i][2])

#Convert the values into integers
values = np.rint(percentile)


song_df_1 = pd.DataFrame()
user = pd.Series(user_id)
song = pd.Series(song_id)
percentile = pd.Series(values)

song_df_1['user_id'] = user.values
song_df_1['song_id'] = song.values
song_df_1['listen_count'] = percentile.values

#Fetch the songs data
song_df_2 =  pd.read_csv('song_data.csv')

#Merge the songs data and the user data into one dataframe
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
song_df = song_df.drop(["title", "release"],1)
print(song_df.columns)

#Type conversion
for col in song_df.select_dtypes(include=['object']).columns:
    print("Converting")
    song_df[col] = song_df[col].astype('category')
    

for col in song_df.select_dtypes(include=['category']).columns:
    print("Converting")
    song_df[col] = song_df[col].cat.codes

#Choose listen count as the target class
target = song_df.pop('listen_count')
#Apply cross validation on the data
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(song_df, target, test_size = 0.3)
del song_df

#Define the model
model = LinearRegression()

model.fit(train_data, train_labels)


#Save the model for evaluation
pickle.dump(model, open("LR_baseline.pickle.dat", "wb"))

predict_labels = model.predict(train_data)

mse_error_train = mean_squared_error(train_labels, predict_labels, multioutput='raw_values')

#Train error 
print("Training error:",mse_error_train)


predict_labels = model.predict(test_data)

mse_error_test = mean_squared_error(test_labels, predict_labels, multioutput='raw_values')

#Test error
print("Testing error:",mse_error_test)


