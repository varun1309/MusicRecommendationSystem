import pandas
from scipy import stats
from sklearn import cross_validation, grid_search, metrics, ensemble
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
import numpy as np

triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_1 = pandas.read_table('10000.txt',header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#print(song_df_1.head())


'''
def calculate_percentile(user,song_df_percentile,index_start,index_end,list_listen_val):

    for i in range(index_start,index_end+1):
        print(user, song_df.iloc[i]['user_id'])
        current_listen_count=song_df.iloc[i]['listen_count']
        percentile_val=stats.percentileofscore(list_listen_val, current_listen_count)
        if percentile_val >= 75:
            song_df_percentile.iloc[i]['listen_count']=4
        elif percentile_val >= 50 and percentile_val<75:
            song_df_percentile.iloc[i]['listen_count'] = 3
        elif percentile_val >= 25 and percentile_val<50:
            song_df_percentile.iloc[i]['listen_count'] = 2
        else:
            song_df_percentile.iloc[i]['listen_count'] = 1
    return song_df_percentile
    
'''

#Read song  metadata
song_df_2 =  pandas.read_csv('song_data.csv')

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
#song_df_1=song_df_1[0:10000]
song_df_1=song_df_1.sort_values('user_id')

#print("FIRST:",song_df_1.head())

last_user=song_df_1.iloc[0]['user_id']
list_listen_val=[]
index_start=0
index_end=0
for i in range (len(song_df_1)):
    #print(row)
    if last_user == song_df_1.iloc[i]['user_id']:
        list_listen_val.append(song_df_1.iloc[i]['listen_count'])
        #
        continue
    index_end=i
    # calculate percentile
    #song_df_1=calculate_percentile(song_df_1.iloc[i-1]['user_id'],song_df_1,index_start,index_end,list_listen_val)
    for i in range(index_start,index_end+1):
        current_listen_count=song_df.iloc[i]['listen_count']
        user = song_df.iloc[i]['user_id']
        percentile_val=stats.percentileofscore(list_listen_val, current_listen_count)
        if percentile_val >= 75:
            song_df_1.iloc[i, song_df_1.columns.get_loc('listen_count')] = 4
            #song_df_1.loc[i,'listen_count']=4
            #song_df_1.iloc[i]['listen_count']=4
        elif percentile_val >= 50 and percentile_val<75:
            song_df_1.iloc[i, song_df_1.columns.get_loc('listen_count')] = 3
        elif percentile_val >= 25 and percentile_val<50:
            song_df_1.iloc[i, song_df_1.columns.get_loc('listen_count')] = 2
        else:
            song_df_1.iloc[i, song_df_1.columns.get_loc('listen_count')] = 1


    list_listen_val=[]
    index_start=index_end+1

    last_user = song_df_1.iloc[i]['user_id']

song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
song_df = song_df.drop(["title", "release"],1)
song_df.to_pickle("./songs_data.pkl")
song_df1 = song_df.to_pickle("./songs_data_.pkl")
print(song_df1)
print(song_df.columns)

for col in song_df.select_dtypes(include=['object']).columns:
    song_df[col] = song_df[col].astype('category')
    
# Encoding categorical features
for col in song_df.select_dtypes(include=['category']).columns:
    song_df[col] = song_df[col].cat.codes


target = song_df.pop('listen_count')
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(song_df, target, test_size = 0.3)
del song_df

#model = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=250)
model = LinearRegression()
model.fit(train_data, train_labels)
#print(test_data)
#predict_labels = model.predict(test_data)
#print(predict_labels)
acc = model.score(test_data, test_labels)
print(acc*100,'%')
print(model.predict(test_data))

y_pred = model.predict(test_data) 
plt.plot(test_data, test_labels, '.')

# plot a line, a perfit predict would all fall on this line
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()



#print(metrics.classification_report(test_labels, predict_labels))
#unique_users_count = len(song_df_1[0].unique())
#unique_songs_count = len(song_df_1[1].unique())




pass
#for each user identify all songs listened calculate score for each song
#
