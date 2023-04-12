# %matplotlib inline
import os, pickle
import pandas as pd
import numpy as np
from copy import deepcopy
# np.set_printoptions(threshold=30)

# Amazon_Books: rating:22507154  user:8026324  movie: 2330066 | Filtered: ratings:3010555 user:103428 item:57324 train_rating: 2400000 test_rating:610555 |threshold:5 ratings:9713850 user: 806542 item:413158 train_rating:8000000 test_rating: 1713850
# Amazon_CD: ratings:3749003 user:1578596 movie:486360 | Filtered: ratings:643455 user:39545 item:24379 train_rating: 500000 test_rating:143455 | threshold:5 ratings:1245489 user:110782 item:74945 train_rating:1000000 test_rating:245489

def remove_infrequent_items(data, min_counts=5):
    df = deepcopy(data)
    counts = df['itemID'].value_counts()
    df = df[df["itemID"].isin(counts[counts >= min_counts].index)]
    print("items with < {} interactoins are removed".format(min_counts))
    return df


def remove_infrequent_users(data, min_counts=10):
    df = deepcopy(data)
    counts = df['userID'].value_counts()
    df = df[df["userID"].isin(counts[counts >= min_counts].index)]
    print("users with < {} interactoins are removed".format(min_counts))
    return df

def generate_inverse_mapping(data_list):
    inverse_mapping = dict()
    for inner_id, true_id in enumerate(data_list):
        inverse_mapping[true_id] = inner_id
    return inverse_mapping

Train_Size = 8000000    #2400000 500000  8000000 1000000
Output_Dim = 1           # 1 5
user_num = 806542        #103428 39545 806542  110782
threshold = 5           #20 10 5 5
DATA_SET_NAME = 'Amazon_Books' # 'Amazon_Books' 'Amazon_CD' 
DATA_PATH     = '../data'
ratings        = pd.read_csv(os.path.join(DATA_PATH, DATA_SET_NAME, 'ratings_Books.csv' if DATA_SET_NAME == 'Amazon_Books' else 'ratings_CDs_and_Vinyl.csv'),index_col=None, encoding='utf-8')

names = ['userID', 'itemID', 'rating', 'timesteps']
ratings.columns = names

print('')
print('The number of ratings: {}'.format(ratings.count()['itemID']))

print('')
print('min value of rating: {}'.format(ratings['rating'].min()))
print('max value of rating: {}'.format(ratings['rating'].max()))

print('')
ra = ratings.groupby(ratings['userID']).count()
print('The number of user in ratings.csv: {}'.format(ra.count()[0]))
print('The minimum number of ratings per user in ratings.csv: {}'.format(ra['itemID'].min()))
print('The maximun number of ratings per user in ratings.csv: {}'.format(ra['itemID'].max()))

print('')
ra = ratings.groupby(ratings['itemID']).count()
print('The number of movies in ratings.csv: {}'.format(ra.count()[0]))
print('The minimum number of ratings per movie in ratings.csv: {}'.format(ra['userID'].min()))
print('The maximun number of ratings per movie in ratings.csv: {}'.format(ra['userID'].max()))


print('filtering users...')
ratings = remove_infrequent_users(ratings, threshold)
print('filtering items...')
ratings = remove_infrequent_items(ratings, threshold)

data = ratings.groupby('userID')['itemID'].apply(list)
#unique_data = ratings.groupby('userID')['itemID'].nunique()
#data = data[unique_data[unique_data >= 10].index]

user_item_dict = data.to_dict()
user_mapping = []
item_set = set()
for user_id, item_list in data.iteritems():
    user_mapping.append(user_id)
    for item_id in item_list:
        item_set.add(item_id)
item_mapping = list(item_set)

print('Filtered num of users:{}, num of items:{}'.format(len(user_mapping), len(item_mapping)))

user_inverse_mapping = generate_inverse_mapping(user_mapping)
item_inverse_mapping = generate_inverse_mapping(item_mapping)


ratings.userID   = np.vectorize(lambda i:user_inverse_mapping[i] )(ratings.userID) 
ratings.itemID   = np.vectorize(lambda i:item_inverse_mapping[i] )(ratings.itemID) 

remove_fields = ['timesteps','rating']
target = ratings['rating']
feature  = ratings.drop(remove_fields, axis=1)

features = feature.values
targets  = target.values

targets   = 1 * np.vectorize(lambda i: i > 3.5)(targets) if Output_Dim == 1 else np.floor(targets-0.5)

from sklearn.model_selection import train_test_split
train_features, test_features, train_target, test_target = train_test_split(features, targets, train_size = Train_Size, random_state = 0, shuffle=True)
print('\ntrain_features\n', train_features, type(train_features), train_features.shape)
print('\ntrain_target\n',   train_target,   type(train_target),   train_target.shape)
print('\ntest_features\n',  test_features,  type(test_features),  test_features.shape)
print('\ntest_target\n',    test_target,    type(test_target),    test_target.shape)

user_count_dict   = dict()
user_rating_count = list()
for line in train_features[:,0]:
    if line in user_count_dict:
        user_count_dict[line] += 1
        user_rating_count.append([user_count_dict[line]])
    else:
        user_count_dict[line] = 1
        user_rating_count.append([1])


movie_count_dict = dict()
movie_rating_count = list()
for line in train_features[:,1]:
    if line in movie_count_dict:
        movie_count_dict[line] += 1
        movie_rating_count.append([movie_count_dict[line]])
    else:
        movie_count_dict[line] = 1
        movie_rating_count.append([1])

train_feature_data = pd.DataFrame(train_features, columns=list(feature.columns))
train_feature_data['user_frequency']  = pd.DataFrame(user_rating_count)
train_feature_data['movie_frequency'] = pd.DataFrame(movie_rating_count)
train_feature_data.to_csv('./{}/{}_train_feature_data_{}_{}_{}.csv'.format(DATA_PATH, DATA_SET_NAME, Train_Size, Output_Dim, threshold), index = None)
print('train_feature_data_{}_{}_{}.csv done.\n'.format(Train_Size, Output_Dim, threshold))

user_count_dict   = dict()
user_rating_count = list()
for line in test_features[:,0]:
    if line in user_count_dict:
        user_count_dict[line] += 1
        user_rating_count.append([user_count_dict[line]])
    else:
        user_count_dict[line] = 1
        user_rating_count.append([1])

movie_count_dict   = dict()
movie_rating_count = list()
for line in test_features[:,1]:
    if line in movie_count_dict:
        movie_count_dict[line] += 1
        movie_rating_count.append([movie_count_dict[line]])
    else:
        movie_count_dict[line] = 1
        movie_rating_count.append([1])

test_feature_data = pd.DataFrame(test_features, columns=list(feature.columns))
test_feature_data['user_frequency']  = pd.DataFrame(user_rating_count)
test_feature_data['movie_frequency'] = pd.DataFrame(movie_rating_count)
test_feature_data.to_csv('./{}/{}_test_feature_data_{}_{}_{}.csv'.format(DATA_PATH, DATA_SET_NAME, len(test_feature_data), Output_Dim, threshold), index = None)
print('test_feature_data_{}_{}_{}.csv done.\n'.format(len(test_feature_data),Output_Dim, threshold))

train_features = train_feature_data.values
test_features  = test_feature_data.values

print("train_features:", train_feature_data.head())
print("train_targets:", train_target[:10])

print('\ntrain_features\n', train_features, type(train_features), train_features.shape)
print('\ntrain_target\n',   train_target,   type(train_target),   train_target.shape)
print('\ntest_features\n',  test_features,  type(test_features),  test_features.shape)
print('\ntest_target\n',    test_target,    type(test_target),    test_target.shape)
pickle.dump((train_features, test_features, train_target, test_target), open('./{}/{}_TrainTest_{}_{}_{}.data'.format(DATA_PATH, DATA_SET_NAME, Train_Size, Output_Dim, threshold), 'wb'))

dict_t = {}
dict_t['userId'] = test_features[:,0]
dict_t['movieId'] = test_features[:,1]
pd_data = pd.DataFrame.from_dict(dict_t)
user_test = pd_data.groupby(pd_data['userId']).count().count()[0]
print('{}% users in test set ({} users)'.format(round(user_test/user_num*100, 2), user_test))

dict_t = {}
dict_t['userId'] = train_features[:,0]
dict_t['movieId'] = train_features[:,1]
pd_data = pd.DataFrame.from_dict(dict_t)
user_train = pd_data.groupby(pd_data['userId']).count().count()[0]
print('{}% users in training set ({} users)'.format(round(user_train/user_num*100, 2), user_train))