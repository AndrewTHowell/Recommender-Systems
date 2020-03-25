# Section: Import modules

from RS import Recommender

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

    print(userRatings)

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

        stringDF = renamedDF.to_string(index=False,
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


# Section End

# Section: Main UI program

class RecommendationUI:

    def __init__(self):
        self.RS = Recommender(train=False)

    def run(self):
        self.running = True
        while self.running:
            self.login()

            self.loggedIn = True
            while self.loggedIn:
                self.setMood()

                self.moodSet = True
                while self.moodSet:
                    self.menu()

    def login(self):
        print("\n\n*** Login page ***\n")

        self.userID = None
        self.userIDs = self.RS.userIDs()
        while self.userID is None or self.userID not in self.userIDs:
            self.userID = int(input("Enter User ID: "))
        print("\nLogin successful!")
        print(f"Welcome back User {self.userID}")

    def setMood(self):
        print("\n\n*** Mood Menu ***")
        print("\nWhat current mood are you in?\n")
        print("Happy")
        print("Sad")
        print("Active")
        print("Lazy\n")
        self.mood = None
        while self.mood not in ["happy", "sad", "active", "lazy"]:
            self.mood = input("Enter mood: ").lower()
        print("\nMood set")
        print(f"You are feeling {self.mood}")

    def menu(self):
        print("\n\n*** Main Menu ***")
        print("1. Get recommendation")
        print("2. Edit user profile")
        print("7. Change mood")
        print("8. Logout")
        print("9. Exit\n")

        menuChoice = input("Enter choice: ")

        if menuChoice == "1":
            self.recommendation()

        elif menuChoice == "2":
            #editProfile(userID, books, ratings)
            print("Editted profile")

        elif menuChoice == "7":
            self.moodSet = False

        elif menuChoice == "8":
            self.moodSet = False
            self.loggedIn = False

        elif menuChoice == "9":
            self.moodSet = False
            self.loggedIn = False
            self.running = False

    def recommendation(self):
        recommendationSize = input("\nHow many recommended music tracks "
                                   "would you like? (default is 5): ")
        if recommendationSize.isdigit():
            recommendationSize = int(recommendationSize)
        else:
            recommendationSize = 5

        recommendedMusic = self.RS.recommendation(self.userID, self.mood,
                                                  recommendationSize)

        print(f"\nBased on your previous ratings, here are your "
              "{recommendationSize} recommended music tracks:")
        counter = 1
        for musicTrack in recommendedMusic:
            title = musicTrack["title"]
            artist = musicTrack["artist"]
            print(f"{counter}. {title} by {artist}")
            counter += 1


# Section End

if __name__ == "__main__":
    RSUI = RecommendationUI()
    RSUI.run()
