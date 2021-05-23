import abc
from typing import Tuple
import pandas as pd
import numpy as np


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)


    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        RMSE = 0
        for index, row in true_ratings.iterrows():
            RMSE += (row['rating'] - self.predict(row['user'], row['item'], row['timestamp'])) ** 2
        RMSE /= len(true_ratings.index)
        return np.sqrt(RMSE)


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.all_movies_AVG = ratings["rating"].mean()
        self.ratings_data = ratings.copy()
        self.ratings_data['rating'] -= self.all_movies_AVG
        self.user_means = self.ratings_data.groupby('user')['rating'].mean()
        self.movie_means = self.ratings_data.groupby('item')['rating'].mean()
        print("hi")

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        predicted_rating = self.all_movies_AVG + self.user_means[user] + self.movie_means[item]
        if predicted_rating < 0.5:
            return 0.5
        if predicted_rating > 5:
            return 5
        return predicted_rating


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        pass


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        pass


class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass
