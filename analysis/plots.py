"""
Plottings
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from data_preprocessing import *


def num_of_users_voting(ratings : rec_data):
    """
    plots number of users voting for each movies
    """
    no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
    f,ax = plt.subplots(1,1,figsize=(16,4))
    plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
    plt.axhline(y=10,color='r')
    plt.xlabel('MovieId')
    plt.ylabel('No. of users voted')
    plt.show()
 


def num_of_votes_by_users(data : rec_data):
    """
    plots number of votes for each user
    """
    no_movies_voted = data.ratings.groupby('userId')['rating'].agg('count') 
    f,ax = plt.subplots(1,1,figsize=(16,4))
    plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
    plt.axhline(y=50,color='r')
    plt.xlabel('UserId')
    plt.ylabel('No. of votes by user')
    plt.show()
    


def plot_count_by_genre(movies: rec_data):
    """
    Plots the bar for Genres
    """
    plt.figure(figsize=(20,7))
    generlist = movies['genres'].apply(lambda generlist_movie : str(generlist_movie).split("|"))
    geners_count = {}

    for generlist_movie in generlist:
        for gener in generlist_movie:
            if(geners_count.get(gener,False)):
                geners_count[gener]=geners_count[gener]+1
            else:
                geners_count[gener] = 1       
    geners_count.pop("(no genres listed)")
    plt.bar(geners_count.keys(),geners_count.values(),color='m')
    plt.show()




 




 
