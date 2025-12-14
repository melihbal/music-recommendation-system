# Part 1: Conditional Probability Modeling

"""
Part 1: Conditional Probability Modeling

Estimate how likely a song is to receive a 5★ rating using conditional probabilities.
"""

import pandas as pd
import numpy as np



def load_global_data():
    tracks = pd.read_csv("../data/tracks.csv")
    ratings = pd.read_csv("../data/ratings.csv")

    # merge the tracks and ratings file based on tracj_id = song_id, to see everything on the same file
    df = ratings.merge(tracks, left_on="song_id", right_on="track_id")

    #add extra column to see if the song is rated as 5 star
    df["is_5star"] = (df["rating"] == 5).astype(int)

    return df


def load_gulces_personal_data():
    tracks = pd.read_csv("../data/tracks.csv")
    gulce_ratings = pd.read_csv("../data/gulce_ratings.csv")

    df = gulce_ratings.merge(tracks, left_on="song_id", right_on="track_id")
    df["is_5star"] = (df["rating"] == 5).astype(int)
    return df


def load_melihs_personal_data():
    tracks = pd.read_csv("../data/tracks.csv")
    melih_ratings = pd.read_csv("../data/melih_ratings.csv")

    df = melih_ratings.merge(tracks, left_on="song_id", right_on="track_id")
    df["is_5star"] = (df["rating"] == 5).astype(int)
    return df



def p_5_given(df, feature, value, alpha=1):
    """
    Computes P(5★ | feature = value) with Laplace smoothing
    """

    #this subset just takes the rows with given feature
    subset = df[df[feature] == value]

    #total ratings with given feature
    total = len(subset)
    if total == 0:
        return None
    
    #number of 5 star ratings & given features count
    num_5 = subset["is_5star"].sum()
    
    # conditional probability = (5 star & feature) / feature with laplace
    return (num_5 + alpha) / (total + 2 * alpha)


def p_5_given_two(df, feature1, value1, feature2, value2, alpha=1):
    """
    P(5★ | feature1=value1, feature2=value2) with Laplace smoothing
    """
    # subset for feature combinations
    subset = df[
        (df[feature1] == value1) &
        (df[feature2] == value2)
    ]

    total = len(subset)
    if total == 0:
        return None

    num_5 = subset["is_5star"].sum()
    return (num_5 + alpha) / (total + 2 * alpha)


def bayes(df, feature, value):
    """
    Computes P(feature=value | 5★) using Bayes' rule
    """
    # P(5★ | feature)
    p_5_given_f = p_5_given(df, feature, value)
    if p_5_given_f is None:
        return None

    # P(feature)
    p_f = len(df[df[feature] == value]) / len(df)

    # P(5★)
    p_5 = df["is_5star"].mean()

    return (p_5_given_f * p_f) / p_5




def main():

    #Load tracks.csv and ratings.csv
    global_df = load_global_data()
    gulce_df= load_gulces_personal_data()
    #melih_df= load_melihs_personal_data()

    #Compute P(5★ | Artist), P(5★ | Year), P(5★ | Popularity)
    # some conditional probabilities with one feature
    print("P(5★ | Artist = Doja Cat):",
          p_5_given(global_df, "primary_artist_name", "Doja Cat"))

    print("P(5★ | Year = 2014):",
          p_5_given(global_df, "album_release_year", 2016))
    
    print("P(5★ | danceability: danceable):",
          p_5_given(global_df, "ab_danceability_value", "danceable"))
    
    print("P(5★ | genre = pop):",
          p_5_given(global_df, "ab_genre_rosamerica_value", "pop"))

    print("P(5★ | Explicit = True):",
          p_5_given(global_df, "explicit", True))
    

    # some conditional probabilities with two features
    print("P(5★ | Artist = The Chainsmokers & release year = 2016):",
          p_5_given_two(global_df,"primary_artist_name","The Chainsmokers","album_release_year", 2016))

    print("P(5★ | danceability: danceable & genre = pop):",
          p_5_given_two(global_df,"ab_danceability_value","danceable","ab_genre_rosamerica_value", "pop"))
    
    #bayes rule implementation
    print("P(Artist = Halsey | 5★):",
      bayes(global_df, "primary_artist_name", "Halsey"))

    print("P(Artist = Taylor Swift | 5★):",
      bayes(global_df, "primary_artist_name", "Taylor Swift"))
    



    # GULCE'S PROB
    print("P(5★ | Artist = Doja Cat):",
          p_5_given(gulce_df, "primary_artist_name", "Doja Cat"))
    
    print("P(5★ | genre = pop):",
          p_5_given(gulce_df, "ab_genre_rosamerica_value", "pop"))

    print("P(5★ | danceability: danceable):",
          p_5_given(gulce_df, "ab_danceability_value", "danceable"))
    


    # MELIH'S PROB
    #print("P(5★ | Artist = Doja Cat):",
    #      p_5_given(melih_df, "primary_artist_name", "Doja Cat"))
    
    #print("P(5★ | genre = pop):",
    #      p_5_given(melih_df, "ab_genre_rosamerica_value", "pop"))

    #print("P(5★ | danceability: danceable):",
    #      p_5_given(melih_df, "ab_danceability_value", "danceable"))


    # GROUP PROB = total prob / 2 for the same feature
    # group prob = (p_5_given(melih_df, "primary_artist_name", "Doja Cat") + p_5_given(gulce_df, "primary_artist_name", "Doja Cat")) / 2



if __name__ == "__main__":
    main()
