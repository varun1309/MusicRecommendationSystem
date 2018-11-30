import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats


import numpy as np
import h5py

def save_obj(obj, name ):
    np.save(name+'.npy', obj)

data_to_write = np.random.random(size=(100,20))

# print(songs_summary.head()[['title', 'song_id']])

song_df_1 = pd.read_table("10000.txt", header=None)
#reducing size for testing, original results were shared for complete data
song_df_1=song_df_1[:10000]


print(song_df_1.shape)


n_u = len(song_df_1[0].unique())
n_m = len(song_df_1[1].unique())

sparsity = len(song_df_1)/(n_u*n_m)

train_data, test_data = train_test_split(song_df_1, test_size=0.25)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)




vectorizer = CountVectorizer()
vectorizer.fit_transform(song_df_1[0].unique())
user_dict = vectorizer.vocabulary_

vectorizer1 = CountVectorizer()
vectorizer1.fit_transform(song_df_1[1].unique())
song_dict = vectorizer1.vocabulary_

save_obj(user_dict,'user_dict')
save_obj(song_dict,'song_dict')
# takes training matrix and returns sparse matrix with song ratings of 0-4, using pencentiled frequency
# currently takes time , might need to optimise or create matrix and save to file
def freq_percentile(training_matrix):
    sum_matrix=training_matrix.sum(axis=1)
    for i in range(training_matrix.shape[0]):
        if sum_matrix[i]!=0 :
            training_matrix[i] = training_matrix[i] / sum_matrix[i]
    percentile_matrix = np.zeros((training_matrix.shape[0], training_matrix.shape[1]))
    for i in range(training_matrix.shape[0]):
        temp1=training_matrix[i][np.nonzero(training_matrix[i])]
        for j in range(training_matrix.shape[1]):

            temp=training_matrix[i][j]
            if training_matrix[i][j] !=0:
                percentile__val=stats.percentileofscore(temp1,training_matrix[i][j])
                if percentile__val>=75 and percentile__val<=100:
                    percentile_matrix[i][j]= 3.0 + (percentile__val-75)/25
                elif percentile__val>=50 and percentile__val<75:
                    percentile_matrix[i][j]= 2.0 + (percentile__val-50)/25
                elif percentile__val>=25 and percentile__val<50:
                    percentile_matrix[i][j]= 1.0 + (percentile__val-0)/25
                elif percentile__val>=0 and percentile__val<25:
                    percentile_matrix[i][j]= 0.0 + (percentile__val-0)/25

    return percentile_matrix


# Creating training matrix
training_matrix = np.zeros((n_u, n_m))

for line in train_data.itertuples():
    training_matrix[user_dict[line[1].lower()], song_dict[line[2].lower()]] = line[3]

# Creating test matrix
testing_matrix = np.zeros((n_u, n_m), dtype=np.float64)
training_matrix_new = np.zeros((n_u, n_m), dtype=np.float64)
for line in test_data.itertuples():
    testing_matrix[user_dict[line[1].lower()], song_dict[line[2].lower()]] = line[3]

# training_matrix = np.true_divide(training_matrix, training_matrix.sum(axis=1, keepdims=True))
#saving intermediate calcultion when dataset is large , it can be reused
training_matrix=freq_percentile(training_matrix)
testing_matrix-freq_percentile(testing_matrix)
with h5py.File('training_freq_matrix.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset-training",  data=training_matrix)
with h5py.File('testing_freq_matrix.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset-testing",  data=testing_matrix)





# rmse between original ratings and  two lower dimension matrices
def calc_rmse(ratings, music, user):
    bool_non_zero = ratings != 0  # Indicator  which is zero for missing rating or listen counts
    mean_sq_error = (bool_non_zero * (ratings - np.dot(user, music.T)))**2 # real and predicted listen count error

    return np.sqrt(np.sum(mean_sq_error)/np.sum(bool_non_zero))  # sum of squared errors then square root


# Latent factor method
factor_latent = 20  # Number of latent factor pairs
reg_lambda = 0.8  # Regularisation strength
gamma_l_rate = 0.0002  # Learning rate
num_epochs = 30  # Number of loops through training data
user_matrix = 3 * np.random.rand(n_u, factor_latent)  # Latent factor for users
music_matrix = 3 * np.random.rand(n_m, factor_latent)  # Latent factor for music

# Stochastic Gradient descent
training_errors = []
testing_errors = []
users, items = training_matrix.nonzero()
for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for u, i in zip(users, items):
        error = training_matrix[u, i] - np.dot(user_matrix[u, :], music_matrix[i, :].T)  # Error for current observation

        user_matrix[u, :] += gamma_l_rate * (error * music_matrix[i, :] - reg_lambda * user_matrix[u, :])  # Update user matrix

        music_matrix[i, :] += gamma_l_rate * (error * user_matrix[u, :] - reg_lambda * music_matrix[i, :])  # Update music atrix

    training_errors.append(calc_rmse(training_matrix, music_matrix, user_matrix))  # Training RMSE for this pass
    testing_errors.append(calc_rmse(testing_matrix, music_matrix, user_matrix))  # Test RMSE for this pass

output_matrix = np.dot(user_matrix, music_matrix.T)

sorted_output_matrix=output_matrix.argsort(axis=1)
save_output=np.zeros((sorted_output_matrix.shape[0],11),dtype=object)
output_df=pd.DataFrame()
for index,row in enumerate(sorted_output_matrix):
    user = next(key for key, value in user_dict.items() if value == index)
    save_output[index][0] = user
    for k in range(1,11):

        key = next(key for key, value in song_dict.items() if value == row[k-1])

        save_output[index][k]=key


song_df_metadata =  pd.read_csv('song_data.csv')
#saves top ten predicted songs
np.savetxt('output.txt',save_output,delimiter=",",fmt='%s')

#shows top ten predicted songs for each user, optional can be commented out.
'''
for row in save_output:
    print('User ID: ',row[0])
    songs_rows=row[1:11]
    for k in range(songs_rows.shape[0]):
        row1=songs_rows[k]
        df = song_df_metadata.loc[song_df_metadata['song_id'] == songs_rows[k].upper()]['title']
        df2 = song_df_metadata.loc[song_df_metadata['song_id'] == songs_rows[k].upper()]['artist_name']
'''




# performance plot rmse training vs test
figure, label = plt.subplots()
label.plot(training_errors, color="g", label='Training RMSE')
label.plot(testing_errors, color="r", label='Test RMSE')

label.legend()
plt.show()





