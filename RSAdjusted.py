# Section: Import modules

import pandas as pd

import os
from os.path import dirname, abspath

import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt

from math import sqrt

# Section End

# Section: Constants

DATASETPATH = dirname(abspath(__file__)) + "//Dataset//"

# Section End


# Section: Recommender class

class Recommender():

    def __init__(self, epochs, regularisationLambda, learningRate,
                 context, threshold):
        self.epochs = epochs
        self.regularisationLambda = regularisationLambda
        self.learningRate = learningRate
        self.context = context

        contextualRatings = pd.read_csv(DATASETPATH+"//Contextual Ratings.csv")

        userItemRatings = contextualRatings.groupby(["userID", "itemID"]
                                                    ).mean().reset_index()

        self.originalRatings = userItemRatings.pivot(index="userID",
                                                     columns="itemID",
                                                     values="rating")

        if self.context:
            itemContextRatings = contextualRatings.groupby(["itemID", "mood"]
                                                           ).mean().reset_index()

            self.contextualItems = itemContextRatings.pivot(index="itemID",
                                                            columns="mood",
                                                            values="rating")

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

        # Find mean of all ratings
        contextualMean = np.nanmean(ratings)

        # Find mean of ratings per item
        itemRatingsMean = np.nanmean(ratings, axis=1)
        bItemMeans = itemRatingsMean - contextualMean

        # Find mean of ratings per context
        contextRatingsMean = np.nanmean(ratings, axis=0)
        bContextMeans = contextRatingsMean - contextualMean

        # Adjust ratings
        adjustedRatings = np.nan_to_num(ratings)

        adjustedRatings -= contextualMean

        adjustedRatings = np.transpose(np.transpose(adjustedRatings) - bItemMeans)

        adjustedRatings -= bContextMeans

        self.bContextualItems = pd.DataFrame(adjustedRatings,
                                             index=self.contextualItems.index,
                                             columns=self.contextualItems.columns)

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
                            oldBiRow = self.bContextualItems.loc[itemID]
                            oldBi = np.sum(oldBiRow)
                        else:
                            oldBi = self.bItems.loc[itemID]
                        oldBu = self.bUsers.loc[userID]

                        oldPu = self.P[userIndex]
                        oldQi = self.Q[:, itemIndex]

                        Eiu = self.errorOfPrediction(userID, itemID, oldBi, oldBu)

                        # RMSE
                        RMSE = Eiu ** 2

                        # Length
                        if self.context:
                            BiSquared = np.linalg.norm(oldBiRow) ** 2
                        else:
                            BiSquared = oldBi ** 2
                        BuSquared = oldBu ** 2
                        QiSquared = np.linalg.norm(oldQi) ** 2
                        PuSquared = np.linalg.norm(oldPu) ** 2

                        length = (BuSquared
                                  + BiSquared
                                  + QiSquared
                                  + PuSquared)

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
                            for context in self.bContextualItems.columns:
                                oldBic = oldBiRow.loc[context]
                                newBic = ((self.learningRate
                                           * (Eiu - (self.regularisationLambda
                                                     * oldBic))))
                                self.bContextualItems.loc[itemID][context] = newBic

                        else:
                            newBi = (oldBi + (self.learningRate
                                              * (Eiu
                                                 - (self.regularisationLambda
                                                    * oldBi))))
                            self.bItems.loc[itemID] = newBi

            regularisedRMSEs.append(regularisedRMSE)

        firstRegularisedRMSE = regularisedRMSEs[0]
        RMSE = sqrt((1/ratings) * firstRegularisedRMSE)
        print("\n\nFirst RMSE: {0}".format(RMSE))

        finalRegularisedRMSE = regularisedRMSEs[-1]
        RMSE = sqrt((1/ratings) * finalRegularisedRMSE)
        print("Final RMSE: {0}".format(RMSE))

        plt.plot(regularisedRMSEs)
        plt.show()

        self.generateNormalisedPredictionDF()

    def generateNormalisedPredictionDF(self):
        # Follow the formula ratings formula R=UΣ(V^T)
        predictionMatrix = np.dot(self.P, self.Q)

        # Adjust ratings
        adjustedRatings = predictionMatrix

        adjustedRatings += self.ratingsMean

        adjustedRatings = np.transpose(np.transpose(adjustedRatings) + self.bUsers.values)
        if self.context:
            bItemsM = self.bContextualItems
            bItems = np.sum(bItemsM, axis=1)
        else:
            bItems = self.bItems
        adjustedRatings += bItems.values

        self.predictDF = pd.DataFrame(adjustedRatings,
                                      index=self.originalRatings.index,
                                      columns=self.originalRatings.columns)

        print(self.predictDF)

    def actualRating(self, userID, itemID):
        return self.originalRatings.loc[userID][itemID]

    def predictedRating(self, userID, itemID, Bi, Bu):
        mu = self.ratingsMean

        qiTpu = self.predictionDF.loc[userID][itemID]

        predictedRating = (mu + Bi + Bu + qiTpu)

        return predictedRating

    def errorOfPrediction(self, userID, itemID, Bi, Bu):
        error = (self.actualRating(userID, itemID)
                 - self.predictedRating(userID, itemID, Bi, Bu))

        return error

    def getRecommendation(self, userID, context):
        pass


# Section End


#RS = Recommender(epochs=20, regularisationLambda=0.02, learningRate=0.05, context=False, threshold=0.1)

contextRS = Recommender(epochs=20, regularisationLambda=0.02, learningRate=0.05, context=True, threshold=0.1)

# RS.getRecommendation(1001, "happy")
