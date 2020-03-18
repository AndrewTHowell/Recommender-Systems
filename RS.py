# Section: Import modules

import pandas as pd

import os
from os.path import dirname, abspath

import numpy as np

from scipy.sparse.linalg import svds

# Section End

# Section: Constants

DATASETPATH = dirname(abspath(__file__)) + "//Dataset//"

# Section End


# Section: Recommender class

class Recommender():

    def __init__(self):
        contextualRatings = pd.read_csv(DATASETPATH+"//Contextual Ratings.csv")

        groupedRatings = contextualRatings.groupby(["userID", "itemID"]
                                                   ).mean().reset_index()

        self.R = groupedRatings.pivot(index="userID",
                                      columns="itemID",
                                      values="rating")

        print("self.R")
        print(self.R)

    def train(self):
        # Convert to np array
        ratings = self.R.values
        #print("ratings")
        #print(ratings)

        # Find mean of all ratings
        ratingsMean = np.nanmean(ratings)
        #print("ratingsMean")
        #print(ratingsMean)

        # Find mean of ratings per user
        userRatingsMean = np.nanmean(ratings, axis=1)
        #print("userRatingsMean")
        #print(userRatingsMean)

        userRatingsMeanDF = pd.DataFrame(userRatingsMean,
                                         index=self.R.index,
                                         columns=["Mean Rating"])
        #print("userRatingsMeanDF")
        #print(userRatingsMeanDF)

        # Find mean of ratings per item
        itemRatingsMean = np.nanmean(ratings, axis=0)
        #print("itemRatingsMean")
        #print(itemRatingsMean)

        itemRatingsMeanDF = pd.DataFrame(itemRatingsMean,
                                         index=self.R.columns,
                                         columns=["Mean Rating"])
        #print("itemRatingsMeanDF")
        #print(itemRatingsMeanDF)

        # Convert NaNs to 0
        ratings = np.nan_to_num(ratings)
        #print("ratings")
        #print(ratings)

        # Run Singular Value Decomposition on the matrix
        # U: user features matrix - how much users like each feature
        # Σ: diagonal matrix singular values/weights
        # V^T: music features matrix - how relevant each feature is to each music
        U, sigma, Vt = svds(ratings, k=min(ratings.shape)-1)

        # Reconvert the sum back into a diagonal matrix
        sigma = np.diag(sigma)

        # Dot product of sigma Vt
        sigmaVt = np.dot(sigma, Vt)

        self.P = U
        #print("self.P")
        #print(self.P)
        self.Q = sigmaVt
        #print("self.Q")
        #print(self.Q)

        # Follow the formula ratings formula R=UΣ(V^T)
        allPredictedRatings = np.dot(U, sigmaVt)

        # Convert back to workable DataFrame
        predictionDF = pd.DataFrame(allPredictedRatings,
                                    columns=self.R.columns)

        #print("predictionDF")
        #print(predictionDF)

    def estimate(self, userID, context):

        # Row of items and ratings for user(not contextual)
        # self.R.loc[userID]

        pass

# Section End


RS = Recommender()

RS.train()

"""
# Find mean of ratings
itemRatingsMeanVector = np.mean(R, axis=0)

itemRatingsMeanMatrix = np.repeat(np.array([itemRatingsMeanVector]),
                                  R.shape[0], axis=0)

userBaselineParameterMatrix = np.zeros(R.shape)
userBaselineParameterMatrix.fill(USERBASELINEPARAMETER)
"""
