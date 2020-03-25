# Section: Import modules

import pandas as pd

import os
from os.path import dirname, abspath

import sys

from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt

from math import sqrt

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# Section End

# Section: Constants

DATASETPATH = dirname(abspath(__file__)) + "//Dataset//"

# Section End

# Section: Parameters

# number of latent features
# Improve by minimising Root Mean Square Error
# 10 to minimum of ratings array shape should be good
K = 20

EPOCHS = 20
REGULARISATIONLAMBDA = 0.02
LEARNINGRATE = 0.05
THRESHOLD = 0.1

# Section End

# Section: Recommender class


class Recommender():

    def __init__(self, context=True, train=True):
        self.epochs = EPOCHS
        self.regularisationLambda = REGULARISATIONLAMBDA
        self.learningRate = LEARNINGRATE
        self.context = context
        self.threshold = THRESHOLD

        contextualRatings = pd.read_csv(DATASETPATH+"//Contextual Ratings.csv")

        userItemRatings = contextualRatings.groupby(["userID", "itemID"]
                                                    ).mean().reset_index()

        self.originalRatings = userItemRatings.pivot(index="userID",
                                                     columns="itemID",
                                                     values="rating")

        self.musicTracks = pd.read_csv(DATASETPATH+"//Music.csv", index_col=0)

        if self.context:
            itemContextRatings = contextualRatings.groupby(["itemID", "mood"]
                                                           ).mean().reset_index()

            self.contextualItems = itemContextRatings.pivot(index="itemID",
                                                            columns="mood",
                                                            values="rating")

        if train:
            self.train()

    def setupPredictionDF(self):
        # Convert to np array
        ratings = self.originalRatings.values

        # Find mean of all ratings
        self.ratingsMean = np.nanmean(ratings)

        # Find mean of ratings per user
        userRatingsMean = np.nanmean(ratings, axis=1)

        self.bUsers = pd.Series(userRatingsMean - self.ratingsMean,
                                index=self.originalRatings.index)

        # Find mean of ratings per item
        itemRatingsMean = np.nanmean(ratings, axis=0)

        self.bItems = pd.Series(itemRatingsMean - self.ratingsMean,
                                index=self.originalRatings.columns)

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
        self.Q = Vt

        # Follow the formula ratings formula R=UΣ(V^T)
        self.predictionMatrix = np.dot(U, sigmaVt)

        self.predictionDF = pd.DataFrame(self.predictionMatrix,
                                         index=self.originalRatings.index,
                                         columns=self.originalRatings.columns)

    def setupContextItems(self):
        # Convert to np array
        ratings = self.contextualItems.values

        # Find mean of ratings per user
        itemRatingsMean = np.nanmean(ratings, axis=1)

        self.bContextualItemMeans = pd.Series(itemRatingsMean - self.ratingsMean,
                                              index=self.originalRatings.columns)

        # Find mean of all ratings
        contextualMean = np.nanmean(ratings)

        self.bContextualItems = np.nan_to_num(ratings - contextualMean)

    def train(self):
        self.setupPredictionDF()
        if self.context:
            self.setupContextItems()

        # Stochastic Gradient Descent
        regularisedRMSEs = []
        for epoch in range(self.epochs):
            print("Epoch {0}".format(epoch + 1), end="\r", flush=True)
            regularisedRMSE = 0
            ratings = 0
            for itemID in self.originalRatings.columns:
                itemIndex = self.originalRatings.columns.get_loc(itemID)
                for userID in self.originalRatings.index:
                    userIndex = self.originalRatings.index.get_loc(userID)
                    if not np.isnan(self.originalRatings.loc[userID][itemID]):
                        # Values
                        if self.context:
                            oldBi = self.bContextualItems[itemIndex]
                            sumOldBi = np.nansum(oldBi)
                        else:
                            oldBi = self.bItems.loc[itemID]
                        oldBu = self.bUsers.loc[userID]

                        oldPu = self.P[userIndex]
                        oldQi = self.Q[:, itemIndex]

                        Eiu = self.errorOfPrediction(userID, itemID)

                        # RMSE
                        RMSE = Eiu ** 2

                        # Length
                        if self.context:
                            BiSquared = np.linalg.norm(oldBi) ** 2
                        else:
                            BiSquared = oldBi ** 2
                        BuSquared = oldBu ** 2
                        QiNormSquared = np.linalg.norm(oldQi) ** 2
                        PuNormSquared = np.linalg.norm(oldPu) ** 2

                        length = (BuSquared
                                  + BiSquared
                                  + QiNormSquared
                                  + PuNormSquared)

                        # regularised RMSE
                        ratings += 1
                        regularisedRMSE += (RMSE
                                            + (self.regularisationLambda
                                               * length))

                        # Move Pu along toward minimum
                        newPu = (oldPu + (self.learningRate
                                          * (Eiu * oldQi
                                             - (self.regularisationLambda
                                                * oldPu))))
                        self.P[userIndex] = newPu

                        # Move Qi along toward minimum
                        newQi = (oldQi + (self.learningRate
                                          * (Eiu * oldPu
                                             - (self.regularisationLambda
                                                * oldQi))))
                        self.Q[:, itemIndex] = newQi

                        # Move Bu along toward minimum
                        newBu = (oldBu + (self.learningRate
                                          * (Eiu
                                             - (self.regularisationLambda
                                                * oldBu))))
                        self.bUsers.loc[userID] = newBu

                        # Move Bi along toward minimum
                        if self.context:
                            newBi = ((self.learningRate
                                      * (Eiu - (self.regularisationLambda
                                                * oldBi))))
                            self.bContextualItems[itemIndex] += newBi
                        else:
                            newBi = (oldBi + (self.learningRate
                                              * (Eiu
                                                 - (self.regularisationLambda
                                                    * oldBi))))
                            self.bItems.loc[itemID] = newBi

            regularisedRMSEs.append(regularisedRMSE)

        firstRegularisedRMSE = regularisedRMSEs[0]
        firstRMSE = sqrt((1/ratings) * firstRegularisedRMSE)
        print("\nFirst RMSE: {0}".format(firstRMSE))

        finalRegularisedRMSE = regularisedRMSEs[-1]
        finalRMSE = sqrt((1/ratings) * finalRegularisedRMSE)
        print("Final RMSE: {0}".format(finalRMSE))

        plt.plot(regularisedRMSEs)
        plt.show()

    def actualRating(self, userID, itemID):
        return self.originalRatings.loc[userID][itemID]

    def predictedRating(self, userID, itemID):
        mu = self.ratingsMean

        if self.context:
            itemIndex = self.originalRatings.columns.get_loc(itemID)
            Bis = self.bContextualItems[itemIndex]
            Bi = np.nansum(Bis)
        else:
            Bi = self.bItems.loc[itemID]

        Bu = self.bUsers.loc[userID]

        qiTpu = self.predictionDF.loc[userID][itemID]

        predictedRating = (mu + Bi + Bu + qiTpu)

        return predictedRating

    def errorOfPrediction(self, userID, itemID):
        error = (self.actualRating(userID, itemID)
                 - self.predictedRating(userID, itemID))

        return error

    # Output: [{"title": _, "artist": _},...] ordered by best predicted rating
    def getRecommendation(self, userID, context, size):
        recommendations = []
        for itemID in self.itemIDs():
            predictedRating = self.predictedRating(userID, itemID)
            recommendations.append({"itemID": itemID,
                                    "predictedRating": predictedRating})

        recommendations.sort(key=lambda track: track["predictedRating"],
                             reverse=True)

        recommendations = recommendations[:size]

        for recommendation in recommendations:
            itemID = recommendation["itemID"]
            musicTrack = self.musicTracks.loc[itemID]
            recommendation["title"] = musicTrack["title"]
            recommendation["artist"] = musicTrack["artist"]

        return recommendations

    def userIDs(self):
        return list(self.originalRatings.index.values)

    def itemIDs(self):
        return list(self.originalRatings.columns.values)


# Section End

RS = Recommender(train=False)
