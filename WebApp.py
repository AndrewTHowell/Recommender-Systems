# Section: Import modules

import pandas as pd

from os.path import dirname, abspath

import numpy as np

from scipy.sparse.linalg import svds

from flask import Flask, redirect, url_for, request
from flask import render_template, flash,  session

import os

# Section End

# Section: Constants

CURRENTPATH = dirname(abspath(__file__))

# Section End

# Section: Functions


# Credit to https://beckernick.github.io/matrix-factorization-recommender/
# This function is inspired by this web pages content
def getPredictionDF(music, ratings):
    # Create a pivot table, with users as rows, music as columns,
    # and ratings as cell values
    RDataFrame = ratings.pivot(index="userID",
                               columns="musicID",
                               values="rating").fillna(0)

    # Convert to np array
    R = RDataFrame.values

    # Find mean of ratings
    userRatingsMean = np.mean(R, axis=1)

    # De-mean all values in np array
    demeanedPivot = R - userRatingsMean.reshape(-1, 1)

    # Run Singular Value Decomposition on the matrix
    # U: user features matrix - how much users like each feature
    # Σ: diagonal matrix singular values/weights
    # V^T: music features matrix - how relevant each feature is to each music track
    U, sigma, Vt = svds(demeanedPivot, k=min(demeanedPivot.shape)-1)

    # Reconvert the sum back into a diagonal matrix
    sigma = np.diag(sigma)

    # Follow the formula ratings formula R=UΣ(V^T), adding back on the means
    allPredictedRatings = (np.dot(np.dot(U, sigma), Vt)
                           + userRatingsMean.reshape(-1, 1))

    # Convert back to workable DataFrame
    predictionDF = pd.DataFrame(allPredictedRatings,
                                columns=RDataFrame.columns)

    return predictionDF


# Credit to https://beckernick.github.io/matrix-factorization-recommender/
# This function is inspired by this web pages content
def getRecommendedMusic(userID, music, ratings, predictionDF, recommendSize=5):

    # Retrieve and sort the predicted ratings for the user
    sortedUserPredictions = (predictionDF.iloc[userID]
                             .sort_values(ascending=False))

    ratedMusicInfo = getRatedMusicInfo(userID, music, ratings)

    # Music info from music the user has not rated
    nonRatedMusicInfos = music[(~music['musicID']
                                .isin(ratedMusicInfo['musicID']))]

    # Merge this info with all predictions
    mergedInfo = nonRatedMusicInfos.merge((pd.DataFrame(sortedUserPredictions)
                                           .reset_index()),
                                          how='left',
                                          left_on='musicID',
                                          right_on='musicID')

    # Rename the predictions column from userId to 'Predictions'
    renamedInfo = mergedInfo.rename(columns={userID: 'Predictions'})

    # Sort so best predictions are at the top
    sortedInfo = renamedInfo.sort_values('Predictions', ascending=False)

    # Reduce list down to only show top recommendSize rows and
    recommendedMusic = sortedInfo.iloc[:recommendSize, :-1]

    return recommendedMusic


def getRatedMusicInfo(userID, music, ratings):

    # Get all of the user's ratings
    userRatings = ratings.loc[ratings["userID"] == userID]

    # Join/merge these ratings with the music info
    joinedUserRatings = (userRatings.merge(music, how="left",
                                           left_on='musicID',
                                           right_on='musicID'
                                           ).sort_values(['rating'],
                                                         ascending=False))

    return joinedUserRatings


def editProfile(userID, music, ratings):
    exit = False
    while not exit:
        print("\n*** User {0} Menu ***".format(userID))
        print("\n** Ratings **")
        ratedMusicInfo = getRatedMusicInfo(userID, music, ratings)

        renamedDF = ratedMusicInfo.rename(columns={"musicID": 'Music ID',
                                                   "musicTitle": 'Music Title',
                                                   "musicGenre": 'Music Genres',
                                                   "rating": 'Rating'})
        print(renamedDF)
        stringDF = renamedDF.to_string(index=False, max_rows=10,
                                       columns={"Music ID",
                                                "Music Title",
                                                "Music Genres",
                                                "Rating"})
        print(stringDF)

        print("\n** Options **")
        print("1. Add music rating")
        print("2. Edit music rating")
        print("3. Delete music rating")
        print("9. Return to Main Menu")

        menuChoice = input("\nEnter choice: ")

        if menuChoice == "1" or menuChoice == "2":
            musicID = int(input("Enter music ID: "))
            musicIDsRated = ratedMusicInfo["musicID"].unique()
            if musicID in musicIDsRated:
                currentRatingRow = ratings.loc[(ratings["userID"] == userID)
                                               & (ratings["musicID"] == musicID)]
                currentRating = currentRatingRow.iloc[0]["rating"]
                print("You rated it {0}/5".format(currentRating))
                rating = int(input("Enter rating (0-5): "))
                ratings.loc[(ratings["userID"] == userID)
                            & (ratings["musicID"] == musicID), "rating"] = rating
            else:
                rating = int(input("Enter rating (0-5): "))
                ratings = ratings.append(pd.DataFrame([[userID,
                                                        musicID,
                                                        rating]],
                                                      columns=["userID",
                                                               "musicID",
                                                               "rating"]),
                                         ignore_index=True)

        elif menuChoice == "3":
            musicID = int(input("Enter music ID: "))
            musicIDsRated = userRatings["musicID"].unique()
            if musicID in musicIDsRated:
                currentRatingRow = ratings.loc[(ratings["userID"] == userID)
                                               & (ratings["musicID"] == musicID)]
                currentRating = currentRatingRow.iloc[0]["rating"]
                print("You rated it {0}/5".format(currentRating))
                confirm = input("Confirm delete by typing 'DEL': ").upper()
                if confirm == "DEL":
                    deleteIndex = ratings[(ratings["userID"] == userID)
                                          & (ratings["musicID"] == musicID)].index
                    ratings.drop(deleteIndex, inplace=True)
            else:
                print("You have not rated this music")
        elif menuChoice == "9":
            exit = True


def passwordCorrect(userID, password):
    if userID == int(password):
        return True
    return False


def getMusicHTML():
    music = pd.read_csv(CURRENTPATH+"//music.csv")
    music = music.rename(columns={"musicID": 'ID',
                                  "musicTitle": 'Title',
                                  "musicGenre": 'Genres'})
    vHTML = dfToHTML(music)

    return musicHTML


def dfToHTML(df):
    dfHTML = df.to_html(classes='table table-striped '
                                'table-hover table-responsive',
                        header=True, index=False)
    return dfHTML


# Section End

# Section: Flask App

app = Flask(__name__)
app.secret_key = os.urandom(16)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/music/')
def music():
    return render_template('music.html', tables=[getMusicHTML()])


@app.route('/user/')
def user():
    if "logged_in" in session:
        if session["logged_in"]:
            userID = session["userID"]
            return render_template('user.html')
    else:
        session["logged_in"] = False
        flash("Please log in first")
        return redirect(url_for('index'))


@app.route('/user/recommend/')
def recommend():
    music = pd.read_csv(CURRENTPATH+"//music.csv")
    ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
    userID = session["userID"]

    predictionDF = getPredictionDF(music, ratings)

    recommendedMusic = getRecommendedMusic(userID, music, ratings,
                                           predictionDF)

    df = recommendedMusic.rename(columns={"musicID": 'Music ID',
                                          "musicTitle": 'Music Title',
                                          "musicGenre": 'Music Genres'})

    dfHTML = dfToHTML(df)

    return render_template('recommend.html', tables=[dfHTML])


@app.route('/user/profile')
def profile():
    music = pd.read_csv(CURRENTPATH+"//music.csv")
    ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
    userID = session["userID"]

    ratedMusicInfo = getRatedMusicInfo(userID, music, ratings)

    renamedDF = ratedMusicInfo.rename(columns={"musicID": 'Music ID',
                                               "musicTitle": 'Music Title',
                                               "musicGenre": 'Music Genres',
                                               "rating": 'Rating'})

    dfHTML = renamedDF.to_html(index=False, columns={"Music ID",
                                                     "Music Title",
                                                     "Music Genres",
                                                     "Rating"})

    return render_template('profile.html', tables=[dfHTML])


@app.route('/user/profile/add', methods=["POST", "GET"])
def add():
    if request.method == "GET":
        return render_template('add.html', tables=[getMusicHTML()])
    else:
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        form = request.form
        userID = session["userID"]
        musicID = form["musicID"]
        rating = form["rating"]
        ratings = ratings.append(pd.DataFrame([[userID, musicID, rating]],
                                              columns=["userID",
                                                       "musicID",
                                                       "rating"]),
                                 ignore_index=True)
        ratings.to_csv(CURRENTPATH+"//ratings.csv", index=False)
        flash("Music rating added")
        return redirect(url_for('add'))


@app.route('/user/profile/edit', methods=["POST", "GET"])
def edit():
    if request.method == "GET":
        music = pd.read_csv(CURRENTPATH+"//music.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        userID = session["userID"]

        ratedMusicInfo = getRatedMusicInfo(userID, music, ratings)

        renamedDF = ratedMusicInfo.rename(columns={"musicID": 'Music ID',
                                                   "musicTitle": 'Music Title',
                                                   "musicGenre": 'Music Genres',
                                                   "rating": 'Rating'})

        dfHTML = renamedDF.to_html(index=False, columns={"Music ID",
                                                         "Music Title",
                                                         "Music Genres",
                                                         "Rating"})

        return render_template('edit.html', tables=[dfHTML])
    else:
        music = pd.read_csv(CURRENTPATH+"//music.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        form = request.form
        userID = session["userID"]
        musicID = int(form["musicID"])
        rating = form["rating"]
        ratedMusicInfo = getRatedMusicInfo(userID, music, ratings)
        musicIDsRated = ratedMusicInfo["musicID"].unique()
        if musicID in musicIDsRated:
            print(ratings.loc[(ratings["userID"] == userID)
                              & (ratings["musicID"] == musicID)])
            ratings.loc[(ratings["userID"] == userID)
                        & (ratings["musicID"] == musicID), "rating"] = rating

            ratings.to_csv(CURRENTPATH+"//ratings.csv", index=False)
        else:
            flash("You have not rated this music")

        return redirect(url_for('edit'))


@app.route('/user/profile/delete', methods=["POST", "GET"])
def delete():
    if request.method == "GET":
        music = pd.read_csv(CURRENTPATH+"//music.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        userID = session["userID"]

        ratedMusicInfo = getRatedMusicInfo(userID, music, ratings)

        renamedDF = ratedMusicInfo.rename(columns={"musicID": 'Music ID',
                                                   "musicTitle": 'Music Title',
                                                   "musicGenre": 'Music Genres',
                                                   "rating": 'Rating'})

        dfHTML = renamedDF.to_html(index=False, columns={"Music ID",
                                                         "Music Title",
                                                         "Music Genres",
                                                         "Rating"})

        return render_template('delete.html', tables=[dfHTML])
    else:
        music = pd.read_csv(CURRENTPATH+"//music.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        form = request.form
        userID = session["userID"]
        musicID = int(form["musicID"])
        ratedMusicInfo = getRatedMusicInfo(userID, music, ratings)
        musicIDsRated = ratedMusicInfo["musicID"].unique()
        if musicID in musicIDsRated:
            deleteIndex = ratings[(ratings["userID"] == userID)
                                  & (ratings["musicID"] == musicID)].index
            ratings.drop(deleteIndex, inplace=True)

            ratings.to_csv(CURRENTPATH+"//ratings.csv", index=False)
        else:
            flash("You have not rated this music")

        return redirect(url_for('delete'))


@app.route('/logout/', methods=["POST"])
def logout():
    session["logged_in"] = False
    session["userID"] = None

    flash("Logged out")
    return redirect(url_for('index'))


@app.route('/login/', methods=["POST"])
def login():
    form = request.form
    userID = form["userID"]
    if userID.isdigit():
        userID = int(userID)
        password = form["password"]
        if passwordCorrect(userID, password):
            session["logged_in"] = True
            session["userID"] = userID
            return redirect(url_for('user'))
        else:
            flash("Password rejected")
            return redirect(url_for('index'))
    else:
        flash("Invalid userID")
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(port=80)

# Section End
