# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:48:14 2018
@author: Shagun
"""
import numpy as np
import pandas as pd

#as the column names are not mentioned in the file so first we need to pass 
#them for each csv file. This is done with the details mentioned in read_me file 

#user files
userColumns = ['userId', 'age', 'sex', 'occupation', 'zipCode']
users = pd.read_csv('ml-100k/u.user', sep='|', names=userColumns,encoding='latin-1')
#rating file:
ratingColumns = ['userId', 'movieId', 'rating', 'unixTimestamp']
movieRatings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratingColumns,encoding='latin-1')
#items file:
itemColumns = ['movieId', 'movieTitle' ,'releaseDate','videoReleaseDate', 'IMDbURL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=itemColumns,
encoding='latin-1')

#loading the training and test data files
movieRatings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=ratingColumns, encoding='latin-1')
movieRatings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=ratingColumns, encoding='latin-1')
movieRatings_train.shape, movieRatings_test.shape

#############################################
     ####    Recommender Engine ######
#############################################

# Collaborative Filtering model is used for recommending purpose.Defining number of users and items
noOfUsers = movieRatings.userId.unique().shape[0]
noOfItems = movieRatings.movieId.unique().shape[0]

#Creating user-item matrix to calculate similarity between users and items
matrixData = np.zeros((noOfUsers,noOfItems))
for line in movieRatings.itertuples():
    matrixData[line[1]-1,line[2]-1] = line[3]

#Calculating the user-user and item-item similarity. using Cosine similarity. Using sklearn's pairwise_distance function
from sklearn.metrics.pairwise import pairwise_distances as pdist
userSimilarity = pdist(matrixData, metric='cosine')
ItemSimilarity = pdist(matrixData.T, metric='cosine')

#Making prediction based on these similarities
def predict(movieRatings, similarity, type='user'):
    if type =='user':
        userMeanRating = movieRatings.mean(axis = 1)
        movieRatingsDiff = movieRatings - userMeanRating[:, np.newaxis]            #to give some format to mean_user_movieRatings, np.newaxis has been used
        pred = userMeanRating[:, np.newaxis] + similarity.dot(movieRatingsDiff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = movieRatings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
userPred = predict(matrixData, userSimilarity, type='user')
itemPred = predict(matrixData, ItemSimilarity, type='item')

#Getting recommende/predicted movie list for the first user
userPredictedListSorted = userPred[1,:].argsort()
movielist = []
for i in userPredictedListSorted:
    movielist.append(items.iloc[i][1])
userPredictedListSorted = itemPred[1,:].argsort()
movielistitem = []
for i in userPredictedListSorted:
    movielistitem.append(items.iloc[i][1])
movielistitem = movielistitem[:15]