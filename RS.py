# Section: Import modules

import pandas as pd

import os
from os.path import dirname, abspath

import sys

import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt

from math import sqrt

# Section End

# Section: Constants

DATASETPATH = dirname(abspath(__file__)) + "//Dataset//"

# Section End


# Section: Recommender class

class Recommender():

    def __init__(self, epochs, regularisationLambda, learningRate):
        self.epochs = epochs
        self.regularisationLambda = regularisationLambda
        self.learningRate = learningRate

        contextualRatings = pd.read_csv(DATASETPATH+"//Contextual Ratings.csv")

        userItemRatings = contextualRatings.groupby(["userID", "itemID"]
                                                    ).mean().reset_index()

<<<<<<< HEAD
=======
        itemContextRatings = contextualRatings.groupby(["itemID", "mood"]
                                                       ).mean().reset_index()

>>>>>>> 2608ecb64304cf194af653900689c51fa34aa399
        self.originalRatings = userItemRatings.pivot(index="userID",
                                                     columns="itemID",
                                                     values="rating")

<<<<<<< HEAD
        itemContextRatings = contextualRatings.groupby(["itemID", "mood"]
                                                       ).mean().reset_index()

=======
>>>>>>> 2608ecb64304cf194af653900689c51fa34aa399
        self.contextualItems = itemContextRatings.pivot(index="itemID",
                                                        columns="mood",
                                                        values="rating")

        self.train()

    def train(self):
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

        # Stochastic Gradient Descent
        regularisedRMSEs = []
        for epoch in range(self.epochs):
            print("Epoch {0}".format(epoch + 1))
            regularisedRMSE = 0
            ratings = 0
            for userID in self.originalRatings.index:
                userIndex = self.originalRatings.index.get_loc(userID)
                for itemID in self.originalRatings.columns:
                    itemIndex = self.originalRatings.columns.get_loc(itemID)
                    if not np.isnan(self.originalRatings.loc[userID][itemID]):

                        # Values
                        Eiu = self.errorOfPrediction(userID, itemID)

                        oldBu = self.bUsers.loc[userID]
                        oldBi = self.bItems.loc[itemID]

                        oldPu = self.P[userIndex]
                        oldQi = self.Q[:, itemIndex]

                        # RMSE
                        RMSE = Eiu ** 2

                        # Length
                        BuSquared = oldBu ** 2
                        BiSquared = oldBi ** 2
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

                        if False: #userID == 1001 and itemID == 251:
                            print("\nActual rating: {0}".format(self.actualRating(userID, itemID)))
                            print("Predicted rating: {0}".format(self.predictedRating(userID, itemID)))
                            print("\nRMSE: {0}".format(RMSE))
                            print("BuSquared: {0}".format(BuSquared))
                            print("BiSquared: {0}".format(BiSquared))
                            print("QiNormSquared: {0}".format(QiNormSquared))
                            print("PuNormSquared: {0}".format(PuNormSquared))
                            print("regularisedRMSE: {0}\n\n".format(regularisedRMSE))

                        # Move Pu along toward minimum
                        newPu = (oldPu + (self.learningRate
                                          * (Eiu * oldQi
                                             - (self.regularisationLambda
                                                * oldPu))))

                        # Move Qi along toward minimum
                        newQi = (oldQi + (self.learningRate
                                          * (Eiu * oldPu
                                             - (self.regularisationLambda
                                                * oldQi))))

                        # Update Pu and Qi
                        self.P[userIndex] = newPu
                        self.Q[:, itemIndex] = newQi

                        # Move Bu along toward minimum
                        newBu = (oldBu + (self.learningRate
                                          * (Eiu
                                             - (self.regularisationLambda
                                                * oldBu))))
                        self.bUsers.loc[userID] = newBu

                        # Move Bi along toward minimum
                        newBi = (oldBi + (self.learningRate
                                          * (Eiu
                                             - (self.regularisationLambda
                                                * oldBi))))
                        self.bItems.loc[itemID] = newBi

            regularisedRMSEs.append(regularisedRMSE)

        finalRegularisedRMSE = regularisedRMSEs[-1]
        RMSE = sqrt((1/ratings) * regularisedRMSE)
        print("\nFinal RMSE: {0}".format(RMSE))

        plt.plot(regularisedRMSEs)
        plt.show()

    def actualRating(self, userID, itemID):

        return self.originalRatings.loc[userID][itemID]

    def predictedRating(self, userID, itemID):
        mu = self.ratingsMean
        Bi = self.bItems.loc[itemID]
        Bu = self.bUsers.loc[userID]
        qiTpu = self.predictionDF.loc[userID][itemID]

        predictedRating = (mu + Bi + Bu + qiTpu)

        return predictedRating

    def errorOfPrediction(self, userID, itemID):
        error = (self.actualRating(userID, itemID)
                 - self.predictedRating(userID, itemID))

        return error


# Section End


RS = Recommender(epochs=50, regularisationLambda=0.02, learningRate=0.02)
