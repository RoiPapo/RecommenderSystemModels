import abc
from typing import Tuple
import pandas as pd
import numpy as np
import sklearn
import datetime
from datetime import datetime
from sklearn import metrics
import math


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
        self.all_movies_AVG = ratings["rating"].mean()
        self.ratings_data = ratings.copy()
        self.ratings_data['rating'] = ratings['rating'] - self.all_movies_AVG
        self.Ru = self.ratings_data.pivot(index='user', columns='item', values='rating').to_numpy()
        self.user_means2 = self.ratings_data.groupby('user')['rating'].mean()
        self.movie_means2 = self.ratings_data.groupby('item')['rating'].mean()
        self.NonZero_User_ratings = np.nan_to_num(self.Ru)
        self.correlation_matrix = metrics.pairwise.cosine_similarity(self.NonZero_User_ratings)
        self.baseLine = BaselineRecommender(ratings)
        self.Ri = self.ratings_data.pivot(index='item', columns='user', values='rating').to_numpy()
        self.Ri_after_zeroes = np.nan_to_num(self.Ri)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        potential_users = [int(index) for index in range(len(self.Ri_after_zeroes[int(item)])) if
                           self.Ri_after_zeroes[int(item)][int(index)] != 0 and int(index) != int(user)]
        baseL_prediction = self.baseLine.predict(user, item, timestamp)
        # closestNeighbors = list(np.absolute((self.user_means2 - self.user_means2[user])).sort_values().index[1:4])
        relevent_row = np.absolute(self.correlation_matrix[int(user)])
        relevent_indexes = (-relevent_row).argsort()
        k = 2
        closestNeighbors = []
        for index in relevent_indexes:
            if k < 0:
                break
            if index in potential_users:
                closestNeighbors.append(index)
                k = k - 1

        numerator = 0
        denum = 0
        for neigbor in closestNeighbors:
            if self.correlation_matrix[int(neigbor)][int(user)] != 0:
                numerator += self.user_similarity(int(user), int(neigbor)) * self.Ru[int(neigbor)][
                    int(item)]
                denum += abs(self.user_similarity(int(user), int(neigbor)))
        predicted_rating = baseL_prediction + (numerator / denum)
        if predicted_rating < 0.5:
            return 0.5
        if predicted_rating > 5:
            return 5
        return predicted_rating

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        return self.correlation_matrix[user1, user2]


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.all_movies_AVG = ratings["rating"].mean()
        self.ratings_data = ratings.copy()
        self.ratings_data['rating'] -= self.all_movies_AVG
        self.user_means = self.ratings_data.groupby('user')['rating'].mean()
        self.movie_means = self.ratings_data.groupby('item')['rating'].mean()

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        num_of_users = len(self.user_means)
        num_of_items = len(self.movie_means)
        num_of_rows = self.ratings_data.shape[0]
        times = self.timestamp_to_vector(timestamp)

        if times[0] == 1:
            times[0] = self.beta[int(num_of_items + num_of_users)]
        if times[1] == 1:
            times[1] = self.beta[int(num_of_items + num_of_users + 1)]
        if times[2] == 1:
            times[2] = self.beta[int(num_of_items + num_of_users + 2)]
        predicted_rating = self.all_movies_AVG + self.beta[int(user)] + self.beta[int(num_of_users - 1 + item)] + sum(
            times)

        if predicted_rating < 0.5:
            return 0.5
        if predicted_rating > 5:
            return 5
        return predicted_rating

    def timestamp_to_vector(self, timestamp):
        times = np.zeros(3)
        if 4 <= datetime.fromtimestamp(timestamp).weekday() <= 5:
            times[2] = 1
        if 6 <= datetime.fromtimestamp(timestamp).hour <= 18:
            times[0] = 1
        else:
            times[1] = 1
        return times

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        num_of_users = len(self.user_means)
        num_of_items = len(self.movie_means)
        num_of_rows = self.ratings_data.shape[0]

        X = np.zeros((num_of_rows, num_of_users + num_of_items + 3))
        for i, row in enumerate(self.ratings_data.iterrows()):
            user = int(row[1]["user"])
            movie = int(row[1]["item"])
            X[i][user] = 1
            X[i][num_of_users + movie - 1] = 1
            times = self.timestamp_to_vector(row[1]["timestamp"])
            X[i, num_of_users + num_of_items: num_of_users + num_of_items + 3] = times
        self.beta = np.linalg.lstsq(X, self.ratings_data['rating'], rcond=None)[0]
        return X, self.beta, self.ratings_data['rating'].to_numpy()


class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.all_movies_AVG = ratings["rating"].mean()
        self.ratings_data = ratings.copy()
        self.baseLine = BaselineRecommender(self.ratings_data)
        # self.ls = LSRecommender(self.ratings_data)
        # x, beta, y = self.ls.solve_ls()
        # self.baseLine.initialize_predictor(ratings)
        self.Ru = self.ratings_data.pivot(index='user', columns='item', values='rating').to_numpy()
        for i in range(self.Ru.shape[0]):
            for j in range(self.Ru.shape[1]):
                if not math.isnan(self.Ru[i][j]):
                    self.Ru[i][j] = self.Ru[i][j] - self.baseLine.predict(i, j, 0)
        self.NonZero_centered_User_ratings = np.nan_to_num(self.Ru).T
        self.centered_corrn_matrix = metrics.pairwise.cosine_similarity(self.NonZero_centered_User_ratings)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        predicted_rating = self.baseLine.predict(user, item, timestamp)
        relevent_row = np.absolute(self.centered_corrn_matrix[int(item)])
        relevent_indexes = (-relevent_row).argsort()
        k = 22
        closestNeighbors = []
        for index in relevent_indexes:
            if not math.isnan(self.Ru[int(user)][int(index)]) and index != item:
                if k < 0:
                    break
                closestNeighbors.append(index)
                k = k - 1

        numerator = 0
        denum = 0
        for neigbor in closestNeighbors:
            numerator += self.centered_corrn_matrix[int(item), int(neigbor)] * self.Ru[int(user)][
                int(neigbor)]
            denum += abs(self.centered_corrn_matrix[int(item), int(neigbor)])
        predicted_rating += (numerator / denum)
        if predicted_rating < 0.5:
            return 0.5
        if predicted_rating > 5:
            return 5
        return predicted_rating


