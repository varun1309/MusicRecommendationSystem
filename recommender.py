import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# print(songs_summary.head()[['title', 'song_id']])

song_df_1 = pd.read_table("10000.txt", header=None)

# song_df_2 = pd.read_csv(songs_metadata_file)

print(song_df_1.shape)
song_df_1 = song_df_1[:100]
# for x,y,z in song_df_1:
#     if z == 0:

print(song_df_1.shape)
n_u = len(song_df_1[0].unique())
n_m = len(song_df_1[1].unique())

sparsity = len(song_df_1)/(n_u*n_m)

train_data, test_data = train_test_split(song_df_1, test_size=0.3)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# normalizing the data
# train_data[2] = (train_data[2] - train_data[2].mean()) / train_data[2].std()


vectorizer = CountVectorizer()
vectorizer.fit_transform(song_df_1[0].unique())
user_dict = vectorizer.vocabulary_

vectorizer1 = CountVectorizer()
vectorizer1.fit_transform(song_df_1[1].unique())
song_dict = vectorizer1.vocabulary_

# Creating training matrix
training_matrix = np.zeros((n_u, n_m))
print("----------------")
for line in train_data.itertuples():
    # if line[3] != 0:
    print('user: ',user_dict[line[1].lower()], 'song:', song_dict[line[2].lower()], ' ', line[3])
    training_matrix[user_dict[line[1].lower()], song_dict[line[2].lower()]] = line[3]

# Creating test matrix
testing_matrix = np.zeros((n_u, n_m), dtype=np.float64)
training_matrix_new = np.zeros((n_u, n_m), dtype=np.float64)
for line in test_data.itertuples():
    testing_matrix[user_dict[line[1].lower()], song_dict[line[2].lower()]] = line[3]

# training_matrix = np.true_divide(training_matrix, training_matrix.sum(axis=1, keepdims=True))
for i in range(training_matrix.shape[0]):
    row_sum = np.nansum(training_matrix[i])
    print("i: ", i, "sum: ", row_sum)
    print(training_matrix[i])
    if row_sum != 0:
        training_matrix[i] = (training_matrix[i] / row_sum) * 100;
# training_matrix = training_matrix_new
print(np.argmax(training_matrix, axis=1))
# testing_matrix = np.true_divide(testing_matrix, testing_matrix.sum(axis=1, keepdims=True))



# Scoring Function: Root Mean Squared Error
def rmse_score(R, Q, P):
    I = R != 0  # Indicator function which is zero for missing data
    ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
    MSE = ME**2
    return np.sqrt(np.sum(MSE)/np.sum(I))  # sum of squared errors


# Set parameters and initialize latent factors
f = 5  # Number of latent factor pairs
lmbda = 0.5  # Regularisation strength
gamma = 0.01  # Learning rate
n_epochs = 5  # Number of loops through training data
P = 3 * np.random.rand(n_u, f)  # Latent factors for users
Q = 3 * np.random.rand(n_m, f)  # Latent factors for movies

# Stochastic GD
train_errors = []
test_errors = []
users, items = training_matrix.nonzero()
for epoch in range(n_epochs):
    print("Epoch:", epoch)
    for u, i in zip(users, items):
        e = training_matrix[u, i] - np.dot(P[u, :], Q[i, :].T)  # Error for this observation
        print(e)
        P[u, :] += gamma * (e * Q[i, :] - lmbda * P[u, :])  # Update this user's features
        print("P: " , max(P[u]))
        Q[i, :] += gamma * (e * P[u, :] - lmbda * Q[i, :])  # Update this movie's features
        print("Q: ", max(Q[i]))
    train_errors.append(rmse_score(training_matrix, Q, P))  # Training RMSE for this pass
    test_errors.append(rmse_score(testing_matrix, Q, P))  # Test RMSE for this pass

output_matrix = np.dot(P, Q.T)
print(output_matrix[0])
# Print how long it took

# Check performance by plotting train and test errors
fig, ax = plt.subplots()
ax.plot(train_errors, color="g", label='Training RMSE')
ax.plot(test_errors, color="r", label='Test RMSE')
# snp.labs("Number of Epochs", "RMSE", "Error During Stochastic GD")
ax.legend()
