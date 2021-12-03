# Movie Recommender System

This is a movie recommender system that is based on the following algorithm.
* Content Based Filtering
* Item-Based Collaborative Filtering with SVD
* Hybrid Recommender - (Content Based + SVD)

This movie recommender is built based on [MovieLens Latest 100k Dataset](https://grouplens.org/datasets/movielens/latest/). 

Example of script output:
```
--------------------------------------------------
Example of Movies Watched by user 1 are:

Star Wars: Episode VI - Return of the Jedi (1983)
Space Jam (1996)
Escape to Witch Mountain (1975)
--------------------------------------------------
--------------------------------------------------
Content Filtering

Top 5 Recommended Movies for user 1 are:

Monday (2000)
White Ribbon, The (Das weiße Band) (2009)
Many Adventures of Winnie the Pooh, The (1977)
Beerfest (2006)
Spartacus (1960)
--------------------------------------------------
--------------------------------------------------
Item Based Colaborative Filtering

Top 5 Recommended Movies for user 1 are:

Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
Monty Python and the Holy Grail (1975)
Lock, Stock & Two Smoking Barrels (1998)
Office Space (1999)
Schindler's List (1993)
--------------------------------------------------
--------------------------------------------------
Hybrid Recommender

Top 5 Recommended Movies for user 1 are:

Princess Bride, The (1987)
North by Northwest (1959)
Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
Life Is Beautiful (La Vita è bella) (1997)
In Bruges (2008)
--------------------------------------------------
```
## Files

* [EDA.ipynb](https://github.com/hwchua0209/Recommender-System/blob/master/EDA.ipynb) - Exploratory Data Analysis of the Dataset

* [main.py](https://github.com/hwchua0209/Recommender-System/blob/master/main.py) - Main python file for running of Movie Recommender System

* [recommender.py](https://github.com/hwchua0209/Recommender-System/blob/master/recommender.py) - Code for recommender system class

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure the correct environment is created

```
conda activate <env>
conda install pip
pip freeze > requirements.txt
```

### Run Recommender

The recommender can be run by using the following command

```
python3 main.py --userid 1
```
Feel free to change userid from (1 to 610) to see movies recommendation for different user.

## Future Improvements

* Model deployment to Heroku with Flask
* Build recommender with Tensorflow TFRS