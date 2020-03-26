# Section: Import modules

from RS import Recommender

# Section End

# Section: Functions


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
            self.giveRecommendation()

        elif menuChoice == "2":
            self.profileMenu()

        elif menuChoice == "7":
            self.moodSet = False

        elif menuChoice == "8":
            self.moodSet = False
            self.loggedIn = False

        elif menuChoice == "9":
            self.moodSet = False
            self.loggedIn = False
            self.running = False

    def giveRecommendation(self):
        recommendationSize = input("\nHow many recommended music tracks "
                                   "would you like? (default is 5): ")
        if recommendationSize.isdigit():
            recommendationSize = int(recommendationSize)
        else:
            recommendationSize = 5

        recommendedMusic = self.RS.getRecommendation(self.userID, self.mood,
                                                     recommendationSize)

        print(f"\nBased on your previous ratings, here are your "
              "{recommendationSize} recommended music tracks:")
        counter = 1
        for musicTrack in recommendedMusic:
            title = musicTrack["title"]
            artist = musicTrack["artist"]
            print(f"{counter}. {title} by {artist}")
            counter += 1

    def profileMenu(self):
        exitProfile = False
        while not exitProfile:
            print(f"\n\n*** User {self.userID} Menu ***")
            print("\n** Ratings **")

            userRatings = self.RS.getUserRatings(self.userID, self.mood)
            for userRating in userRatings:
                trackTitle = userRating["track"]["title"]
                trackArtist = userRating["track"]["artist"]
                rating = userRating["rating"]
                print(f"You rated {trackTitle} by {trackArtist},"
                      f" {rating} out of 5")

            print("\n** Options **")
            print("1. Add book rating")
            print("2. Edit book rating")
            print("3. Delete book rating")
            print("9. Return to Main Menu")

            menuChoice = input("\nEnter choice: ")

            if menuChoice == "1":
                self.addRating()

            elif menuChoice == "2":
                self.changeRating()

            elif menuChoice == "3":
                self.deleteRating()

            elif menuChoice == "9":
                exitProfile = True

    def addRating(self):
        itemID = int(input("Enter the music track ID: "))
        rating = int(input("Enter rating (0-5): "))
        self.RS.addRating(self.userID, itemID, self.mood, rating)

    def changeRating(self):
        itemID = int(input("Enter the music track ID: "))
        currentRating = self.RS.getUserRating(self.userID, itemID, self.mood)
        currentRating = currentRating["rating"]
        print(f"Current rating is {currentRating}")

        self.RS.deleteRating(self.userID, itemID, self.mood)
        rating = int(input("Enter new rating (0-5): "))
        self.RS.addRating(self.userID, itemID, self.mood, rating)

    def deleteRating(self):
        itemID = int(input("Enter the music track ID: "))
        self.RS.deleteRating(self.userID, itemID, self.mood)

# Section End


if __name__ == "__main__":
    RSUI = RecommendationUI()
    RSUI.run()
