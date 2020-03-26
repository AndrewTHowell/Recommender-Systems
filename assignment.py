# Section: Import modules

from RS import Recommender

from math import inf as infinity

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

        print("Goodbye")

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

        moods = ["happy", "sad", "active", "lazy"]
        self.mood = self.getChoice("Enter mood", moods)["text"]

        print("\nMood set")
        print(f"You are feeling {self.mood}")

    def menu(self):
        print("\n\n*** Main Menu ***\n")

        choices = ["Get recommendation",
                   "Edit user profile",
                   "Change mood",
                   "Logout",
                   "Exit"]
        while True:
            choice = self.getChoice("Enter choice", choices)["num"]

            if choice == 1:
                self.giveRecommendation()

            elif choice == 2:
                self.profileMenu()

            elif choice == 3:
                self.moodSet = False
                break

            elif choice == 4:
                self.moodSet = False
                self.loggedIn = False
                break

            elif choice == 5:
                self.moodSet = False
                self.loggedIn = False
                self.running = False
                break

    def giveRecommendation(self):
        message = "How many recommended music tracks would you like?"
        recommendationSize = self.getNum(message, 1, 30, 5)

        recommendedMusic = self.RS.getRecommendation(self.userID, self.mood,
                                                     recommendationSize)

        print(f"\nBased on your previous ratings, here are your "
              f"{recommendationSize} recommended music tracks:")
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
            print("\n** Ratings **\n")

            userRatings = self.RS.getUserRatings(self.userID, self.mood)
            for userRating in userRatings:
                trackID = userRating["track"]["itemID"]
                trackTitle = userRating["track"]["title"]
                trackArtist = userRating["track"]["artist"]
                rating = userRating["rating"]
                print(f"You rated {trackTitle} by {trackArtist},"
                      f" {rating} out of 5 (Track ID: {trackID})")
            if userRatings == []:
                print("You haven't provided any ratings in this mood yet")

            print("\n** Options **\n")

            choices = ["Add book rating",
                       "Edit book rating",
                       "Delete book rating",
                       "Return to Main Menu"]

            choice = self.getChoice("Enter choice", choices)["num"]

            if choice == 1:
                self.addRating()

            elif choice == 2:
                self.changeRating()

            elif choice == 3:
                self.deleteRating()

            elif choice == 4:
                exitProfile = True

    def addRating(self):
        itemID = int(input("Enter the music track ID: "))
        rating = self.getNum("Enter rating", 0, 5)
        self.RS.addRating(self.userID, itemID, self.mood, rating)

    def changeRating(self):
        itemID = int(input("Enter the music track ID: "))
        currentRating = self.RS.getUserRating(self.userID, itemID, self.mood)

        if currentRating is not None:
            currentRating = currentRating["rating"]
            print(f"Current rating is {currentRating}")

            self.RS.deleteRating(self.userID, itemID, self.mood)
            rating = self.getNum("Enter rating", 0, 5)
            self.RS.addRating(self.userID, itemID, self.mood, rating)

        else:
            print("You have not rated this music track before")
            input("Press any key to continue")

    def deleteRating(self):
        itemID = int(input("Enter the music track ID: "))
        self.RS.deleteRating(self.userID, itemID, self.mood)

    def getNum(self, message, minimum=-infinity, maximum=infinity, default=False):
        num = None
        while num not in range(minimum, maximum):
            minText = ""
            if minimum != -infinity:
                minText = f" (minimum: {minimum})"

            maxText = ""
            if maximum != infinity:
                maxText = f" (maximum: {maximum})"

            defaultText = ""
            if default:
                defaultText = f" (default: {default})"

            num = input(f"{message}{minText}{maxText}{defaultText}: ")

            if num == "":
                return default

            if num.isdigit():
                num = int(num)

        return int(num)

    def getChoice(self, message, choices):
        choices = [choice.lower() for choice in choices]

        choiceRange = range(1, len(choices)+1)
        for choiceIndex in choiceRange:
            print(f"{choiceIndex}. {choices[choiceIndex-1].capitalize()}")
        print()

        while True:
            choice = input(f"{message}: ")

            if choice.isdigit():
                choiceIndex = int(choice)
                if choiceIndex in choiceRange:
                    return {"num": choiceIndex,
                            "text": choices[choiceIndex-1]}

            if choice.lower() in choices:
                return {"num": choices.index(choice)+1,
                        "text": choice}


# Section End


if __name__ == "__main__":
    RSUI = RecommendationUI()
    RSUI.run()
