import pandas as pd
import random

from pathlib import Path
from recommender import Content_filtering, Colab_filtering, Hybrid

path = Path('data')


def merge_data():

    movies = pd.read_csv(path / 'movies.csv')
    ratings = pd.read_csv(path / 'ratings.csv')

    df = pd.merge(movies, ratings, on='movieId', how='left')

    # Remove unwanted data
    df.dropna(inplace=True)
    df.drop(columns='timestamp', inplace=True)
    df.drop(index=df[df.genres == '(no genres listed)'].index, inplace=True)

    # Save to new dataset
    df.to_csv(path / 'dataset_final.csv', index=False, encoding='utf-8')


def read_data():

    return pd.read_csv(path / 'dataset_final.csv')


def model(userid):

    merge_data()
    data = read_data()

    # Recommendor
    recommendor1 = Content_filtering(data)
    movie_list, recommended_movie = recommendor1.recommend(userid)

    movie_user = random.sample(movie_list, k=5)
    recommendor_cont = random.sample(recommended_movie, k=5)

    recommendor2 = Colab_filtering(data)
    recommendor_colab = recommendor2.recommend(userid)

    recommendor3 = Hybrid(data)
    recommendor_hybrid = recommendor3.recommend(userid)

    return movie_user, recommendor_cont, recommendor_colab, recommendor_hybrid
