# Movie Recommender System with Flask Web App Deployment

This is a movie recommender system that is based on the following algorithm.
* Content Based Filtering
* Item-Based Collaborative Filtering with SVD
* Hybrid Recommender - (Content Based + SVD)

This movie recommender is built based on [MovieLens Latest 100k Dataset](https://grouplens.org/datasets/movielens/latest/). 

The movie recommeder is then deploy on Flask Web App with RestAPI. 

![flaskapp](/image/Flask%20App%20Output.png)

## Files

* [EDA.ipynb](https://github.com/hwchua0209/Recommender-System/blob/master/EDA.ipynb) - Exploratory Data Analysis of the Dataset

* [main.py](https://github.com/hwchua0209/Recommender-System/blob/master/main.py) - Main python file for feature engineering and running of Movie Recommender System

* [recommender.py](https://github.com/hwchua0209/Recommender-System/blob/master/recommender.py) - Code for recommender system class

* [app.py](https://github.com/hwchua0209/Recommender-System/blob/master/app.py) - Code for Flask App Deployment

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure the correct environment is created

```
conda create --name <env>
conda activate <env>
pip install -r requirements.txt
```

### Run Recommender

The recommender can be run by using the following command

```
python3 app.py
```

## Future Improvements

* Build recommender with Tensorflow TFRS