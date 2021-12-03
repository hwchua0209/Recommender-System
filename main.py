import pandas as pd
import random
import argparse

from pathlib import Path
from recommender import Content_filtering, Colab_filtering, Hybrid

path = Path('data')

parser = argparse.ArgumentParser(description='Recommender System')
parser.add_argument('--userid', default='1',type=int, help='UserID (1 to 610)')

def merge_data():
    
    movies = pd.read_csv(path / 'movies.csv')
    ratings = pd.read_csv(path / 'ratings.csv')

    df = pd.merge(movies,ratings, on='movieId', how='left')

    # Remove unwanted data
    df.dropna(inplace=True)
    df.drop(columns='timestamp', inplace=True)
    df.drop(index=df[df.genres == '(no genres listed)'].index, inplace=True)

    # Save to new dataset
    df.to_csv(path / 'dataset_final.csv', index=False, encoding='utf-8')

def read_data():

    return pd.read_csv(path / 'dataset_final.csv')

def main():

    args = parser.parse_args()

    merge_data()
    data = read_data()

    # Recommendor 
    recommendor1 = Content_filtering(data)
    movie_list, recommended_movie = recommendor1.recommend(args.userid)
    movie_user = random.sample(movie_list, k=3)
    recommendor_cont = random.sample(recommended_movie, k=5)

    recommendor2 = Colab_filtering(data)
    recommendor_colab = recommendor2.recommend(args.userid)

    recommendor3 = Hybrid(data)
    recommendor_hybrid = recommendor3.recommend(args.userid)

    # Display Results on Terminal
    print('-'*50)
    print(f'Example of Movies Watched by user {args.userid} are:\n')
    print(*movie_user, sep='\n')
    print('-'*50)

    print('-'*50)
    print('Content Filtering\n')
    print(f'Top 5 Recommended Movies for user {args.userid} are:\n')
    print(*recommendor_cont, sep='\n')
    print('-'*50)

    print('-'*50)
    print('Item Based Colaborative Filtering\n')
    print(f'Top 5 Recommended Movies for user {args.userid} are:\n')
    print(*recommendor_colab, sep='\n')
    print('-'*50)

    print('-'*50)
    print('Hybrid Recommender\n')
    print(f'Top 5 Recommended Movies for user {args.userid} are:\n')
    print(*recommendor_hybrid, sep='\n')
    print('-'*50)

if __name__ == '__main__':
    main()