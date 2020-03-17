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
def getPredictionDF(books, ratings):
    # Create a pivot table, with users as rows, books as columns,
    # and ratings as cell values
    RDataFrame = ratings.pivot(index="userID",
                               columns="bookID",
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

    return predictionDF


# Credit to https://beckernick.github.io/matrix-factorization-recommender/
# This function is inspired by this web pages content
def getRecommendedBooks(userID, books, ratings, predictionDF, recommendSize=5):

    # Retrieve and sort the predicted ratings for the user
    sortedUserPredictions = (predictionDF.iloc[userID]
                             .sort_values(ascending=False))

    ratedBooksInfo = getRatedBookInfo(userID, books, ratings)

    # Book info from books the user has not rated
    nonRatedBookInfos = books[(~books['bookID']
                               .isin(ratedBooksInfo['bookID']))]

    # Merge this info with all predictions
    mergedInfo = nonRatedBookInfos.merge((pd.DataFrame(sortedUserPredictions)
                                          .reset_index()),
                                         how='left',
                                         left_on='bookID', right_on='bookID')

    # Rename the predictions column from userId to 'Predictions'
    renamedInfo = mergedInfo.rename(columns={userID: 'Predictions'})

    # Sort so best predictions are at the top
    sortedInfo = renamedInfo.sort_values('Predictions', ascending=False)

    # Reduce list down to only show top recommendSize rows and
    recommendedBooks = sortedInfo.iloc[:recommendSize, :-1]

    return recommendedBooks


def getRatedBookInfo(userID, books, ratings):

    # Get all of the user's ratings
    userRatings = ratings.loc[ratings["userID"] == userID]

    # Join/merge these ratings with the book info
    joinedUserRatings = (userRatings.merge(books, how="left",
                                           left_on='bookID',
                                           right_on='bookID'
                                           ).sort_values(['rating'],
                                                         ascending=False))

    return joinedUserRatings


def editProfile(userID, books, ratings):
    exit = False
    while not exit:
        print("\n*** User {0} Menu ***".format(userID))
        print("\n** Ratings **")
        ratedBooksInfo = getRatedBookInfo(userID, books, ratings)

        renamedDF = ratedBooksInfo.rename(columns={"bookID": 'Book ID',
                                                   "bookTitle": 'Book Title',
                                                   "bookGenre": 'Book Genres',
                                                   "rating": 'Rating'})
        print(renamedDF)
        stringDF = renamedDF.to_string(index=False, max_rows=10,
                                       columns={"Book ID",
                                                "Book Title",
                                                "Book Genres",
                                                "Rating"})
        print(stringDF)

        print("\n** Options **")
        print("1. Add book rating")
        print("2. Edit book rating")
        print("3. Delete book rating")
        print("9. Return to Main Menu")

        menuChoice = input("\nEnter choice: ")

        if menuChoice == "1" or menuChoice == "2":
            bookID = int(input("Enter book ID: "))
            bookIDsRated = ratedBooksInfo["bookID"].unique()
            if bookID in bookIDsRated:
                currentRatingRow = ratings.loc[(ratings["userID"] == userID)
                                               & (ratings["bookID"] == bookID)]
                currentRating = currentRatingRow.iloc[0]["rating"]
                print("You rated it {0}/5".format(currentRating))
                rating = int(input("Enter rating (0-5): "))
                ratings.loc[(ratings["userID"] == userID)
                            & (ratings["bookID"] == bookID), "rating"] = rating
            else:
                rating = int(input("Enter rating (0-5): "))
                ratings = ratings.append(pd.DataFrame([[userID,
                                                        bookID,
                                                        rating]],
                                                      columns=["userID",
                                                               "bookID",
                                                               "rating"]),
                                         ignore_index=True)

        elif menuChoice == "3":
            bookID = int(input("Enter book ID: "))
            bookIDsRated = userRatings["bookID"].unique()
            if bookID in bookIDsRated:
                currentRatingRow = ratings.loc[(ratings["userID"] == userID)
                                               & (ratings["bookID"] == bookID)]
                currentRating = currentRatingRow.iloc[0]["rating"]
                print("You rated it {0}/5".format(currentRating))
                confirm = input("Confirm delete by typing 'DEL': ").upper()
                if confirm == "DEL":
                    deleteIndex = ratings[(ratings["userID"] == userID)
                                          & (ratings["bookID"] == bookID)].index
                    ratings.drop(deleteIndex, inplace=True)
            else:
                print("You have not rated this book")
        elif menuChoice == "9":
            exit = True


def passwordCorrect(userID, password):
    if userID == int(password):
        return True
    return False


def getBooksHTML():
    books = pd.read_csv(CURRENTPATH+"//books.csv")
    books = books.rename(columns={"bookID": 'ID',
                                  "bookTitle": 'Title',
                                  "bookGenre": 'Genres'})
    booksHTML = dfToHTML(books)

    return booksHTML


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


@app.route('/books/')
def books():
    return render_template('books.html', tables=[getBooksHTML()])


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
    books = pd.read_csv(CURRENTPATH+"//books.csv")
    ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
    userID = session["userID"]

    predictionDF = getPredictionDF(books, ratings)

    recommendedBooks = getRecommendedBooks(userID, books, ratings,
                                           predictionDF)

    df = recommendedBooks.rename(columns={"bookID": 'Book ID',
                                          "bookTitle": 'Book Title',
                                          "bookGenre": 'Book Genres'})

    dfHTML = dfToHTML(df)

    return render_template('recommend.html', tables=[dfHTML])


@app.route('/user/profile')
def profile():
    books = pd.read_csv(CURRENTPATH+"//books.csv")
    ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
    userID = session["userID"]

    ratedBooksInfo = getRatedBookInfo(userID, books, ratings)

    renamedDF = ratedBooksInfo.rename(columns={"bookID": 'Book ID',
                                               "bookTitle": 'Book Title',
                                               "bookGenre": 'Book Genres',
                                               "rating": 'Rating'})

    dfHTML = renamedDF.to_html(index=False, columns={"Book ID",
                                                     "Book Title",
                                                     "Book Genres",
                                                     "Rating"})

    return render_template('profile.html', tables=[dfHTML])


@app.route('/user/profile/add', methods=["POST", "GET"])
def add():
    if request.method == "GET":
        return render_template('add.html', tables=[getBooksHTML()])
    else:
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        form = request.form
        userID = session["userID"]
        bookID = form["bookID"]
        rating = form["rating"]
        ratings = ratings.append(pd.DataFrame([[userID, bookID, rating]],
                                              columns=["userID",
                                                       "bookID",
                                                       "rating"]),
                                 ignore_index=True)
        ratings.to_csv(CURRENTPATH+"//ratings.csv", index=False)
        flash("Book rating added")
        return redirect(url_for('add'))


@app.route('/user/profile/edit', methods=["POST", "GET"])
def edit():
    if request.method == "GET":
        books = pd.read_csv(CURRENTPATH+"//books.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        userID = session["userID"]

        ratedBooksInfo = getRatedBookInfo(userID, books, ratings)

        renamedDF = ratedBooksInfo.rename(columns={"bookID": 'Book ID',
                                                   "bookTitle": 'Book Title',
                                                   "bookGenre": 'Book Genres',
                                                   "rating": 'Rating'})

        dfHTML = renamedDF.to_html(index=False, columns={"Book ID",
                                                         "Book Title",
                                                         "Book Genres",
                                                         "Rating"})

        return render_template('edit.html', tables=[dfHTML])
    else:
        books = pd.read_csv(CURRENTPATH+"//books.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        form = request.form
        userID = session["userID"]
        bookID = int(form["bookID"])
        rating = form["rating"]
        ratedBooksInfo = getRatedBookInfo(userID, books, ratings)
        bookIDsRated = ratedBooksInfo["bookID"].unique()
        if bookID in bookIDsRated:
            print(ratings.loc[(ratings["userID"] == userID)
                              & (ratings["bookID"] == bookID)])
            ratings.loc[(ratings["userID"] == userID)
                        & (ratings["bookID"] == bookID), "rating"] = rating

            ratings.to_csv(CURRENTPATH+"//ratings.csv", index=False)
        else:
            flash("You have not rated this book")

        return redirect(url_for('edit'))


@app.route('/user/profile/delete', methods=["POST", "GET"])
def delete():
    if request.method == "GET":
        books = pd.read_csv(CURRENTPATH+"//books.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        userID = session["userID"]

        ratedBooksInfo = getRatedBookInfo(userID, books, ratings)

        renamedDF = ratedBooksInfo.rename(columns={"bookID": 'Book ID',
                                                   "bookTitle": 'Book Title',
                                                   "bookGenre": 'Book Genres',
                                                   "rating": 'Rating'})

        dfHTML = renamedDF.to_html(index=False, columns={"Book ID",
                                                         "Book Title",
                                                         "Book Genres",
                                                         "Rating"})

        return render_template('delete.html', tables=[dfHTML])
    else:
        books = pd.read_csv(CURRENTPATH+"//books.csv")
        ratings = pd.read_csv(CURRENTPATH+"//ratings.csv")
        form = request.form
        userID = session["userID"]
        bookID = int(form["bookID"])
        ratedBooksInfo = getRatedBookInfo(userID, books, ratings)
        bookIDsRated = ratedBooksInfo["bookID"].unique()
        if bookID in bookIDsRated:
            deleteIndex = ratings[(ratings["userID"] == userID)
                                  & (ratings["bookID"] == bookID)].index
            ratings.drop(deleteIndex, inplace=True)

            ratings.to_csv(CURRENTPATH+"//ratings.csv", index=False)
        else:
            flash("You have not rated this book")

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
