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

MODELPATH = dirname(abspath(__file__)) + "\\Trained Model\\"
DATASETPATH = dirname(abspath(__file__)) + "\\Dataset\\"

# Section End

# Section: Parameters

# number of latent features
K = 30

EPOCHS = 16
REGULARISATIONLAMBDA = 0.02
LEARNINGRATE = 0.015
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

        self.contextualRatings = pd.read_csv(DATASETPATH
                                             + "//Contextual Ratings.csv")

        userItemRatings = self.contextualRatings.groupby(["userID", "itemID"]
                                                         ).mean().reset_index()

        self.originalRatings = userItemRatings.pivot(index="userID",
                                                     columns="itemID",
                                                     values="rating")

        self.musicTracks = pd.read_csv(DATASETPATH
                                       + "//Music.csv", index_col=0)

        if self.context:
            musicCategories = pd.read_csv(DATASETPATH
                                          + "//Music Categories.csv",
                                          index_col=0)

            itemGenres = pd.merge(self.musicTracks, musicCategories,
                                  left_on="categoryID",
                                  right_index=True,
                                  sort=False)

            genreContextRatings = pd.merge(self.contextualRatings, itemGenres,
                                           on="itemID",
                                           sort=False)

            genreContextRatings = (genreContextRatings
                                   .groupby(["categoryID", "mood"])
                                   .mean().reset_index())

            self.genreContextRatings = (genreContextRatings
                                        .pivot(index="categoryID",
                                               columns="mood",
                                               values="rating"))

        if train:
            self.train()

    def userIDs(self):
        return list(self.originalRatings.index.values)

    def itemIDs(self):
        return list(self.originalRatings.columns.values)

    def train(self):
        if os.path.exists(MODELPATH) and os.path.isdir(MODELPATH):
            # Directory is not empty
            if os.listdir(MODELPATH):
                self.predictionDF = pd.read_csv(MODELPATH + "predictionDF.csv")
                return
        else:
            try:
                os.mkdir(MODELPATH)
            except OSError:
                print("Creation of directory failed")

        self.setupPredictionR()
        if self.context:
            self.setupContextItems()

        # Stochastic Gradient Descent
        RMSEs = []
        for epoch in range(self.epochs):
            print("Epoch {0}".format(epoch + 1))
            errorTotal = 0
            ratings = 0
            for itemID in self.originalRatings.columns:
                itemIndex = self.originalRatings.columns.get_loc(itemID)
                for userID in self.originalRatings.index:
                    userIndex = self.originalRatings.index.get_loc(userID)
                    if not np.isnan(self.originalRatings.loc[userID][itemID]):
                        # Values
                        if self.context:
                            categoryID = self.getTrackInfo(itemID)["categoryID"]
                            oldBi = self.bContextualGenres.loc[categoryID].sum()
                        else:
                            oldBi = self.bItems.loc[itemID]
                        oldBu = self.bUsers.loc[userID]

                        oldPu = self.P[userIndex]
                        oldQi = self.Q[:, itemIndex]

                        Eiu = self.errorOfPrediction(userID, itemID)

                        # Error
                        error = Eiu ** 2

                        # Length
                        BiSquared = oldBi ** 2
                        BuSquared = oldBu ** 2
                        QiNormSquared = np.linalg.norm(oldQi) ** 2
                        PuNormSquared = np.linalg.norm(oldPu) ** 2

                        length = (BuSquared
                                  + BiSquared
                                  + QiNormSquared
                                  + PuNormSquared)

                        # Error
                        ratings += 1
                        errorTotal += error

                        # Move Pu along toward minimum
                        newPu = (oldPu
                                 + (self.learningRate
                                    * ((Eiu * oldQi)
                                       - (self.regularisationLambda * oldPu))))
                        self.P[userIndex] = newPu

                        # Move Qi along toward minimum
                        newQi = (oldQi
                                 + (self.learningRate
                                    * ((Eiu * oldPu)
                                       - (self.regularisationLambda * oldQi))))
                        self.Q[:, itemIndex] = newQi

                        # Move Bu along toward minimum
                        newBu = (oldBu
                                 + (self.learningRate
                                    * (Eiu - (self.regularisationLambda
                                              * oldBu))))
                        self.bUsers.loc[userID] = newBu

                        # Move Bi along toward minimum
                        if self.context:
                            for categoryID in self.bContextualGenres.index:
                                for mood in self.bContextualGenres.columns:
                                    oldBi = (self.bContextualGenres
                                             .loc[categoryID][mood])

                                    newBi = (oldBi
                                             + (self.learningRate
                                                * (Eiu
                                                   - (self.regularisationLambda
                                                      * oldBi))))
                                    (self.bContextualGenres
                                     .loc[categoryID][mood]) = newBi
                        else:
                            newBi = (oldBi
                                     + (self.learningRate
                                        * (Eiu - (self.regularisationLambda
                                                  * oldBi))))
                            self.bItems.loc[itemID] = newBi

            RMSE = sqrt((1/ratings) * errorTotal)
            RMSEs.append(RMSE)

        print("\nFirst RMSE: {0}".format(RMSEs[0]))
        print("Final RMSE: {0}".format(RMSEs[-1]))

        plt.plot(RMSEs)
        plt.show()

        self.generatePredictionDataFrame()

        # Save trained model
        #self.predictionDF.to_csv(MODELPATH + "predictionDF.csv")

    def setupPredictionR(self):
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
        U, sigma, Vt = svds(ratings, k=min(min(ratings.shape)-1, K))

        # Reconvert the sum back into a diagonal matrix
        sigma = np.diag(sigma)

        # Dot product of sigma Vt
        sigmaVt = np.dot(sigma, Vt)

        self.P = U
        self.Q = Vt

        predictionMatrix = np.dot(self.P, self.Q)
        self.predictionR = pd.DataFrame(predictionMatrix,
                                        index=self.originalRatings.index,
                                        columns=self.originalRatings.columns)

    def setupContextItems(self):
        # Convert to np array
        genreContextRatings = self.genreContextRatings.values

        # Find mean of ratings per user
        genreContextRatingsMean = np.nanmean(genreContextRatings)

        bContextualGenres = np.nan_to_num(genreContextRatings
                                          - genreContextRatingsMean)

        self.bContextualGenres = pd.DataFrame(bContextualGenres,
                                              index=(self.genreContextRatings
                                                     .index),
                                              columns=(self.genreContextRatings
                                                       .columns))

    def actualRating(self, userID, itemID):
        return self.originalRatings.loc[userID][itemID]

    def predictedRating(self, userID, itemID):
        mu = self.ratingsMean

        if self.context:
            categoryID = self.getTrackInfo(itemID)["categoryID"]
            Bi = self.bContextualGenres.loc[categoryID].sum()
        else:
            Bi = self.bItems.loc[itemID]

        Bu = self.bUsers.loc[userID]

        qiTpu = self.predictionR.loc[userID][itemID]

        predictedRating = (mu + Bi + Bu + qiTpu)

        return predictedRating

    def generatePredictionDataFrame(self):
        predictionMatrix = np.dot(self.P, self.Q)

        predictionMatrix += self.ratingsMean

        Bu = self.bUsers.values
        predictionMatrix += Bu.reshape(-1, 1)

        if self.context:
            Bi = self.bContextualGenres.values.sum()
        else:
            Bi = self.bItems
        predictionMatrix += Bi

        self.predictionDF = pd.DataFrame(predictionMatrix,
                                         index=self.originalRatings.index,
                                         columns=self.originalRatings.columns)

        print(self.predictionDF)

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

        recommendedTracks = []
        for recommendation in recommendations:
            itemID = recommendation["itemID"]
            recommendedTrack = self.getTrackInfo(itemID)
            recommendedTracks.append(recommendedTrack)

        return recommendedTracks

    def getTrackInfo(self, itemID):
        musicTrack = self.musicTracks.loc[itemID]
        track = {}
        track["itemID"] = itemID
        track["title"] = musicTrack["title"]
        track["artist"] = musicTrack["artist"]
        track["categoryID"] = musicTrack["categoryID"]

        return track

    def getUserRatings(self, userID, mood):
        userRatings = []
        for index, ratingRow in self.contextualRatings.iterrows():
            if ratingRow["userID"] == userID and ratingRow["mood"] == mood:
                userRating = {}
                userRating["context"] = ratingRow["mood"]
                userRating["rating"] = ratingRow["rating"]
                userRating["itemID"] = ratingRow["itemID"]
                trackInfo = self.getTrackInfo(userRating["itemID"])
                userRating["track"] = trackInfo
                userRatings.append(userRating)

        return userRatings

    def getUserRating(self, userID, itemID, mood):
        for index, ratingRow in self.contextualRatings.iterrows():
            if (ratingRow["userID"] == userID
                    and ratingRow["mood"] == mood
                    and ratingRow["itemID"] == itemID):

                userRating = {}
                userRating["context"] = ratingRow["mood"]
                userRating["rating"] = ratingRow["rating"]
                userRating["itemID"] = ratingRow["itemID"]
                userRating["index"] = index
                trackInfo = self.getTrackInfo(userRating["itemID"])
                userRating["track"] = trackInfo
                return userRating

    def addRating(self, userID, itemID, mood, rating):
        df = self.contextualRatings
        df = df.append({"userID": userID,
                        "itemID": itemID,
                        "rating": rating,
                        "mood": mood,
                        },
                       ignore_index=True)
        self.contextualRatings = df

    def deleteRating(self, userID, itemID, mood):
        df = self.contextualRatings
        df.drop(df[(df.userID == userID)
                   & (df.itemID == itemID)
                   & (df.mood == mood)].index, inplace=True)


# Section End

RS = Recommender()
RSc = Recommender(context=False)
