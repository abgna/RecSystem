"""
    testing the data_preprocessing and analysis packages    
"""
from data_preprocessing import *
from analysis import *

# define the data
data = rec_data()
data.read_data()


print(data.shape())
print(data.num_of_movies())
print(data.num_of_null())
print(data.head())

print(data.ratings.head())

num_of_votes_by_users(data)

plot_count_by_genre(data.data)

num_of_users_voting(data.data)

