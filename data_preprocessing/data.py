import os
import numpy as np
import pandas as pd

    """
    Data for recommendation system
    """
    path_movies = '/Users/nareabgaryan/Desktop/n/data/movies.csv'
    path_ratings = "/Users/nareabgaryan/Desktop/n/data/ratings.csv"

    def __init__(self,movies = None,ratings = None,data = None):

        self.movies = movies
        self.ratings = ratings
        self.data = data
        self.movie_rating_thres = 0
        self.user_rating_thres = 0


    def read_data(self):
        """
        Args:
            location of movies and ratings csv
            which should contain movies.csv and ratings.csv
        """
        self.movies = pd.read_csv(self.path_movies)
        self.ratings = pd.read_csv(self.path_ratings)
        #merge the data on given column
        #self.movies = pd.read_csv(fl_name + 'movies.csv')
        #self.ratings = pd.read_csv(fl_name + 'ratings.csv')
        self.data = pd.merge(left = self.movies, right = self.ratings, on = 'movieId')

   
    def shape(self):
        """
        Returns:
            shape: shape
        """ 
return self.data.shape

    def num_of_null(self):
       """null sum of columns
        Returns:
            list: list of integers
        """
        return self.data.isnull().sum()

    def num_of_movies(self,column = 'movieId'):
        """
        returns num of unique values
        Args:
            column (str):  Defaults to 'movieId'.
        Returns:
            list: list of nulls
        """  
        return self.data[column].nunique()


    def head(self):
        """
        Returns:
            DataFrame: head of DataFrame
        """ 
        return self.data.head()

    def set_filter_params(self, movie_rating_thres, user_rating_thres):
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres

    def _prep_data(self):
        """
        prepare data for recommender
        1. movie-user scipy sparse matrix
        2. hashmap of movie to row index in movie-user scipy sparse matrix
        """
        path_ratings = '/Users/nareabgaryan/Desktop/n/data/movies.csv'
        path_movies = "/Users/nareabgaryan/Desktop/n/data/ratings.csv"
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        df_movies = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))  # noqa
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))  # noqa
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        # pivot and create movie-user matrix
        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        # create mapper from movie title to index
        hashmap = {
            movie: i for i, movie in
            enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title)) # noqa
        }
        # transform matrix to scipy sparse matrix
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

 
        return movie_user_mat_sparse, hashmap

def obtain_year(df):
    """return years of publishment

    Args:
        df (rec_data): RecSystem DataFrame
    """
    years = []

    for title in df.data['title']:
        year_subset = title[-5:-1]
        try: years.append(int(year_subset))
        except: years.append(9999)
            
    df.data['moviePubYear'] = years
    #prints values
    print(len(df.data[df.data['moviePubYear'] == 9999]))
    print(df.data[df.data['moviePubYear'] == 9999])

