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

EPOCHS = 30
REGULARISATIONLAMBDA = 0.02
LEARNINGRATE = 0.05

THRESHOLD = 0.2

# Section End

# Section: Recommender class


class Recommender():

    def __init__(self, context=True, retrain=False, showGraphs=True):
        self.epochs = EPOCHS
        self.regularisationLambda = REGULARISATIONLAMBDA
        self.learningRate = LEARNINGRATE
        self.threshold = THRESHOLD

        self.context = context
        self.retrain = retrain
        self.showGraphs = showGraphs

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

        self.train()

# Division: Completed Methods

    def userIDs(self):
        return list(self.originalRatings.index.values)

    def itemIDs(self):
        return list(self.originalRatings.columns.values)

    def train(self):
        if os.path.exists(MODELPATH) and os.path.isdir(MODELPATH):
            # Directory is not empty
            if os.listdir(MODELPATH) and not self.retrain:
                self.predictionDF = pd.read_csv(MODELPATH + "predictionDF.csv",
                                                index_col=0)
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
        MAEs = []
        for epoch in range(self.epochs):
            print("Epoch {0}".format(epoch + 1))
            errorTotal = 0
            errorSquaredTotal = 0
            ratings = 0

            for index, row in self.contextualRatings.iterrows():
                userID = row["userID"]
                userIndex = self.originalRatings.index.get_loc(userID)
                itemID = row["itemID"]
                itemIndex = self.originalRatings.columns.get_loc(itemID)
                rating = row["rating"]
                mood = row["mood"]

                if self.context:
                    categoryID = self.getTrackInfo(itemID)["categoryID"]
                    oldBi = self.bContextualGenres.loc[categoryID][mood]
                else:
                    oldBi = self.bItems.loc[itemID]

                oldBu = self.bUsers.loc[userID]

                oldPu = self.P[userIndex]

                oldQi = self.Q[:, itemIndex]

                Eiu = rating - self.predictedRating(userID, itemID)

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
                    newBi = (oldBi
                             + (self.learningRate
                                * (Eiu
                                   - (self.regularisationLambda
                                      * oldBi))))
                    self.bContextualGenres.loc[categoryID][mood] = newBi
                else:
                    newBi = (oldBi
                             + (self.learningRate
                                * (Eiu - (self.regularisationLambda
                                          * oldBi))))
                    self.bItems.loc[itemID] = newBi

                predictionMatrix = np.dot(self.P, self.Q)
                self.predictionR = pd.DataFrame(predictionMatrix,
                                                index=self.originalRatings.index,
                                                columns=self.originalRatings.columns)

                # Error
                ratings += 1

                error = Eiu
                errorTotal += abs(error)

                errorSquared = error ** 2
                errorSquaredTotal += errorSquared

            RMSE = sqrt((1/ratings) * errorSquaredTotal)
            RMSEs.append(RMSE)

            MAE = errorTotal / ratings
            MAEs.append(MAE)

        print("\nFirst RMSE: {0}".format(RMSEs[0]))
        print("Final RMSE: {0}".format(RMSEs[-1]))

        if self.showGraphs:
            plt.plot(RMSEs)
            plt.show()

        print("\nFirst MAE: {0}".format(MAEs[0]))
        print("Final MAE: {0}".format(MAEs[-1]))

        if self.showGraphs:
            plt.plot(MAEs)
            plt.show()

        self.generatePredictionDataFrame()

        # Save trained model
        self.predictionDF.to_csv(MODELPATH + "predictionDF.csv")

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

    def predictedRating(self, userID, itemID):
        qiTpu = self.predictionR.loc[userID][itemID]

        mu = self.ratingsMean

        Bu = self.bUsers.loc[userID]

        if self.context:
            categoryID = self.getTrackInfo(itemID)["categoryID"]
            Bi = self.bContextualGenres.loc[categoryID].sum()
        else:
            Bi = self.bItems.loc[itemID]

        predictedRating = (qiTpu + mu + Bu + Bi)

        return predictedRating

    def generatePredictionDataFrame(self):
        predictionMatrix = np.dot(self.P, self.Q)

        predictionMatrix += self.ratingsMean

        Bu = self.bUsers.values
        predictionMatrix += Bu.reshape(-1, 1)

        if self.context:
            Bis = []
            for itemID in self.originalRatings.columns:
                categoryID = self.getTrackInfo(itemID)["categoryID"]
                Bi = self.bContextualGenres.loc[categoryID].sum()
                Bis.append(Bi)
            predictionMatrix += Bis
        else:
            Bi = self.bItems
            predictionMatrix += Bi

        self.predictionDF = pd.DataFrame(predictionMatrix,
                                         index=self.originalRatings.index,
                                         columns=self.originalRatings.columns)

    def getRecommendation(self, userID, mood, size):
        if userID not in self.predictionDF.index:
            return []

        recommendations = self.predictionDF.loc[userID]

        recommendations.sort_values(ascending=False, inplace=True)

        #######################################################################
        # Get neighbourhood of users, half the size of the population, using cosine similarity
        contextFocusedRatings = (self.contextualRatings
                                 [self.contextualRatings.mood == mood])

        contextFocusedRatings = contextFocusedRatings.groupby(["userID", "itemID"]
                                                              ).mean().reset_index()

        contextFocusedRatings = contextFocusedRatings.pivot(index="userID",
                                                            columns="itemID",
                                                            values="rating").fillna(0)

        if userID not in contextFocusedRatings.index:
            return []

        similarities = []
        userRow = contextFocusedRatings.loc[userID].values
        userRowNorm = np.linalg.norm(userRow)
        if userRowNorm == 0:
            return []
        for otherUserID, otherUserRow in contextFocusedRatings.iterrows():
            if otherUserID != userID:
                otherUserRow = otherUserRow.values
                dotProduct = np.dot(userRow, otherUserRow)
                otherUserRowNorm = np.linalg.norm(otherUserRow)

                similarity = dotProduct/(userRowNorm * otherUserRowNorm)

                similarities.append({"userID": otherUserID,
                                     "similarity": similarity})

        similarities.sort(key=lambda s: s["similarity"], reverse=True)
        neighbourhood = []
        for similarity in similarities:
            if similarity["similarity"] == 0:
                break
            neighbourhood.append(similarity["userID"])
        neighbourhood = neighbourhood[:len(similarities)//2]

        if len(neighbourhood) == 0:
            return []
        #######################################################################

        recommendedItemIDs = []
        for itemID, value in recommendations.iteritems():
            # Get probability from number of neighbours who rated itemID in mood/total neighbours
            df = self.contextualRatings.query('(itemID == @itemID)'
                                              'and (mood == @mood)'
                                              'and (userID in @neighbourhood)')

            numOfSharedReviews = len(df.index)

            Pk = numOfSharedReviews/len(neighbourhood)

            if Pk >= THRESHOLD:
                recommendedItemIDs.append([int(itemID), Pk])
                if len(recommendedItemIDs) == size:
                    break

        recommendedTracks = []
        for recommendedItemID in recommendedItemIDs:
            recommendedTrack = self.getTrackInfo(recommendedItemID[0])
            recommendedTrack["Pk"] = round(recommendedItemID[1], 2)
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

    def test(self):
        MAE = self.MAE()

        AUPResult = self.accuracyOfUsagePredictionsTest()

        precisionResult = AUPResult["precision"]
        recallResult = AUPResult["recall"]

        print(f"The MAE for this model is {MAE}")

        print(f"The Precision Result for this model is {precisionResult}")

        print(f"The Recall Result for this model is {recallResult}")

    def MAE(self):
        differenceTotal = 0
        for index, row in self.contextualRatings.iterrows():
            actualRating = row["rating"]

            userID = row["userID"]
            itemID = row["itemID"]
            predictedUserRatings = self.predictionDF.loc[userID]

            if isinstance(predictedUserRatings.index[0], np.int64):
                itemID = np.int64(itemID)
            else:
                itemID = str(itemID)

            predictedRating = predictedUserRatings[itemID]

            difference = abs(predictedRating - actualRating)
            differenceTotal += difference

        MAE = differenceTotal / len(self.contextualRatings.index)

        return MAE

# Division End

# Division: Unfinished Methods

    def accuracyOfUsagePredictionsTest(self):
        confusionMatrix = pd.DataFrame(data={"Recommended": [0, 0],
                                             "Not recommended": [0, 0]},
                                       index=["Used", "Not used"])

        for userID in range(1001, 1044):
            for mood in ["happy", "sad", "active", "lazy"]:
                userMoodCount = 0
                for recommendation in RS.getRecommendation(userID, mood, 10):
                    if RS.getUserRating(userID, recommendation["itemID"], mood):
                        userMoodCount += 1
                print(f"User {userID} whilst {mood}: {userMoodCount}")

        precision = (confusionMatrix.loc["Used"]["Recommended"]
                     / (confusionMatrix.loc["Used"]["Recommended"]
                        + confusionMatrix.loc["Not used"]["Recommended"]))

        recall = (confusionMatrix.loc["Used"]["Recommended"]
                  / (confusionMatrix.loc["Used"]["Recommended"]
                     + confusionMatrix.loc["Used"]["Not recommended"]))

        return {"precision": precision, "recall": recall}

# Division End


# Section End

RS = Recommender(context=True, retrain=False, showGraphs=False)

RS.test()
