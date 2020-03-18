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

        #print("self.R")
        #print(self.R)

    def train(self):
        # Convert to np array
        ratings = self.R.values

        # Find mean of all ratings
        self.ratingsMean = np.nanmean(ratings)

        # Find mean of ratings per user
        self.userRatingsMean = np.nanmean(ratings, axis=1)

        self.userRatingsMeanDF = pd.Series(self.userRatingsMean,
                                           index=self.R.index)

        # Find mean of ratings per item
        self.itemRatingsMean = np.nanmean(ratings, axis=0)

        self.itemRatingsMeanDF = pd.Series(self.itemRatingsMean,
                                           index=self.R.columns)

        # Convert NaNs to 0
        ratings = np.nan_to_num(ratings)

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
        self.Q = sigmaVt

        # Follow the formula ratings formula R=UΣ(V^T)
        self.predictionMatrix = np.dot(U, sigmaVt)

        self.predictionDF = pd.DataFrame(self.predictionMatrix,
                                         index=self.R.index,
                                         columns=self.R.columns)


        biasUser = self.userRatingsMeanDF.loc[userID] - mu

        biasItem = self.itemRatingsMeanDF.loc[itemID] - mu

    def originalRating(self, userID, itemID):

        return self.R.loc[userID][itemID]

    def predictedRating(self, userID, itemID):

        return self.predictionDF.loc[userID][itemID]

    def errorOfPrediction(self, userID, itemID):
        error = (self.originalRating(userID, itemID)
                 - self.predictedRating(userID, itemID))

        return error


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
