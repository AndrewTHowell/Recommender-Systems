# Section: Import modules

import pandas as pd

import os
from os.path import dirname, abspath

import numpy as np

from scipy.sparse.linalg import svds
from scipy.linalg import svd

# Section End

# Section: Constants

DATASETPATH = dirname(abspath(__file__)) + "//Dataset//"

# Section End


# Section: Recommender class

class Recommender():

    def __init__(self):
        contextualRatings = pd.read_csv(DATASETPATH+"//Contextual Ratings.csv")

        print("contextualRatings")
        print(contextualRatings)

        groupedRatings = contextualRatings.groupby(["userID", "itemID"]).mean().reset_index()

        print("groupedRatings")
        print(groupedRatings)

        self.R = groupedRatings.pivot(index="userID",
                                      columns="itemID",
                                      values="rating").fillna(0)

        print("self.R")
        print(self.R)

    def train(self):
        # Convert to np array
        self.R = RDataFrame.values

        # Find mean of ratings
        userRatingsMean = np.mean(R, axis=1)

        # De-mean all values in np array
        demeanedPivot = R - userRatingsMean.reshape(-1, 1)

        # Run Singular Value Decomposition on the matrix
        # U: user features matrix - how much users like each feature
        # Σ: diagonal matrix singular values/weights
        # V^T: book features matrix - how relevant each feature is to each book
        U, sigma, Vt = svds(demeanedPivot, k=min(demeanedPivot.shape)-1)

        # Reconvert the sum back into a diagonal matrix
        sigma = np.diag(sigma)

        # Follow the formula ratings formula R=UΣ(V^T), adding back on the means
        allPredictedRatings = (np.dot(np.dot(U, sigma), Vt)
                               + userRatingsMean.reshape(-1, 1))

        # Convert back to workable DataFrame
        predictionDF = pd.DataFrame(allPredictedRatings,
                                    columns=RDataFrame.columns)

    def predict(self, user, context):
        pass

# Section End


RS = Recommender()

"""
# Find mean of ratings
itemRatingsMeanVector = np.mean(R, axis=0)

itemRatingsMeanMatrix = np.repeat(np.array([itemRatingsMeanVector]),
                                  R.shape[0], axis=0)

userBaselineParameterMatrix = np.zeros(R.shape)
userBaselineParameterMatrix.fill(USERBASELINEPARAMETER)
"""
