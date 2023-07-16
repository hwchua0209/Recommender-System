import random
import pandas as pd
import tensorflow as tf
# import tensorflow_recommenders as tfrs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD


class Content_filtering:
    """Content Filtering using tf-idf"""

    def __init__(self, df):
        self.df = df

    def bag_of_words(self):
        """Bag of Words for Content Filtering"""

        data = self.df.groupby(self.df['title'])[
            'genres'].unique().reset_index()
        data['genres'] = data['genres'].apply(str).str[2:-2].str.split('|')

        data['Bag_of_words'] = data['genres'].str.join(' ')
        data = data[['title', 'Bag_of_words']]

        return data

    def tfidf(self, data):
        """TF-idf Vectorizer"""

        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
        tfidf_matrix = tfidf.fit_transform(data['Bag_of_words'])

        # Compute Cosine Similarity of TF-idf Matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        return cosine_sim

    def recommend_genre(self, title):
        """Get Recommendation with Content-Based Filtering"""
        data = self.bag_of_words()

        cosine_sim = self.tfidf(data)

        recommended_movies_genres = []

        indices = pd.Series(data['title'])

        idx = indices[indices == title].index[0]

        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

        top_indices = list(score_series.iloc[1:2].index)

        for i in top_indices:
            recommended_movies_genres.append(list(data['title'])[i])

        return recommended_movies_genres

    def recommend(self, userId):
        """Movies to be Recommended to User Based on Movies User has Watched. """
        recommended_movie = []
        df_filtered = self.df[self.df["userId"] == userId]

        movie_list = df_filtered['title'].tolist()[:10]

        for i in range(len(movie_list)):
            recommended_movie.append(self.recommend_genre(movie_list[i]))
        recommended_movie_final = [y for x in recommended_movie for y in x]

        for movie in recommended_movie_final:
            if movie in movie_list:
                recommended_movie_final.remove(movie)

        return movie_list, recommended_movie_final


class Colab_filtering:
    """Item Based Colaborative Filtering with SVD"""

    def __init__(self, df):
        self.df = df

    def recommend(self, userId):
        """SVD Recommender"""
        reader = Reader()
        data = Dataset.load_from_df(
            self.df[['userId', 'movieId', 'rating']], reader)

        svd_recommender = SVD()
        trainset = data.build_full_trainset()
        svd_recommender.fit(trainset)

        list = self.df[self.df.userId != userId].movieId.unique().tolist()
        predicted_list = []
        for x in list:
            predicted_list.append([x, svd_recommender.predict(userId, x).est])
        df_prediction = pd.DataFrame(predicted_list, columns=[
                                     'movieId', 'Predicted_rating'])
        predict = df_prediction.sort_values(by='Predicted_rating', ascending=False)[
            'movieId'][:5].values
        recommended_movie = []
        for x in predict:
            recommended_movie.append(
                self.df.title[self.df.movieId == x].unique()[0])

        return recommended_movie


class Hybrid:
    """Content Filtering + Item Based Colaborative Filtering"""

    def __init__(self, df):
        self.df = df
        self.cont = Content_filtering(self.df)

    def recommend(self, userId):
        _, recommended_movie = self.cont.recommend(userId)
        data = self.df[self.df.title.isin(recommended_movie)]
        # print(data)
        reader = Reader()
        svd_df = Dataset.load_from_df(
            data[['userId', 'movieId', 'rating']], reader)

        svd_recommender = SVD()
        trainset = svd_df.build_full_trainset()
        svd_recommender.fit(trainset)

        list = data[data.userId != userId].movieId.unique().tolist()
        predicted_list = []
        for x in list:
            predicted_list.append([x, svd_recommender.predict(userId, x).est])
        df_prediction = pd.DataFrame(predicted_list, columns=[
                                     'movieId', 'Predicted_rating'])
        predict = df_prediction.sort_values(by='Predicted_rating', ascending=False)[
            'movieId'][:5].values
        recommended_movie = []
        for x in predict:
            recommended_movie.append(
                self.df.title[self.df.movieId == x].unique()[0])

        return recommended_movie
