# Part 1: Conditional Probability Modeling

"""
Part 1: Conditional Probability Modeling

Estimate how likely a song is to receive a 5★ rating using conditional probabilities.
"""

import pandas as pd
import numpy as np


# DATA LOAD FUNCTIONS

def load_global_data():
    tracks = pd.read_csv("data/tracks.csv")
    ratings = pd.read_csv("data/user_ratings.csv")

    # merge the tracks and ratings file based on tracj_id = song_id, to see everything on the same file
    df = ratings.merge(tracks, left_on="song_id", right_on="track_id")
    return df

def load_gulces_personal_data():
    tracks = pd.read_csv("data/tracks.csv")
    gulce_ratings = pd.read_csv("data/gulce_ratings.csv")
    df = gulce_ratings.merge(tracks, left_on="song_id", right_on="track_id")
    return df

def load_melihs_personal_data():
    tracks = pd.read_csv("data/tracks.csv")
    melih_ratings = pd.read_csv("data/melih_ratings.csv")
    df = melih_ratings.merge(tracks, left_on="song_id", right_on="track_id")
    return df



#CONDITIONAL PROBABILITIES

# Computes P(5★ | feature = value) with Laplace smoothing a=1 b=2
def p_5_given(df, feature, value, alpha=1):

    #this subset just takes the rows with given feature
    subset = df[df[feature] == value]

    #total ratings with given feature
    total = len(subset)
    if total == 0:
        return None
    
    #number of 5 star ratings & given features count
    num_5 = (subset["rating"] == 5).sum()
    
    # conditional probability = (5 star & feature) / feature with laplace
    return (num_5 + alpha) / (total + 2 * alpha)


def p_5_given_two(df, feature1, value1, feature2, value2, alpha=1):
    # subset for feature combinations
    subset = df[
        (df[feature1] == value1) &
        (df[feature2] == value2)
    ]

    total = len(subset)
    if total == 0:
        return None

    num_5 = (subset["rating"] == 5).sum()
    return (num_5 + alpha) / (total + 2 * alpha)



#BAYES THEOREM

# Computes P(feature=value | 5★) using Bayes' rule
def bayes(df, feature, value):

    # P(5★ | feature)
    p_5_given_f = p_5_given(df, feature, value)
    if p_5_given_f is None:
        return None

    # P(feature)
    p_f = len(df[df[feature] == value]) / len(df)

    # P(5★)
    p_5 = (df["rating"] == 5).mean()

    return (p_5_given_f * p_f) / p_5



# TOP FEATURES DUE TO 5 STAR RATINGS

# Computes P(5★ | feature=value) for all values of a feature and returns top_n values with enough data (min_count).
def top_p5_by_feature(df, feature, min_count, top_n=10):
    
    results = []

    for value, group in df.groupby(feature):
        total = len(group)
        if total < min_count:
            continue  #

        num_5 = (group["rating"] == 5).sum()
        p5 = (num_5 + 1) / (total + 2)  # Laplace smoothing
        results.append((value, p5, total))

    return (
        pd.DataFrame(results, columns=[feature, "P(5★)", "count"])
        .sort_values("P(5★)", ascending=False)
        .head(top_n)
    )


if __name__ == "__main__":


    #Load tracks.csv and ratings.csv
    global_df = load_global_data()
    gulce_df= load_gulces_personal_data()
    melih_df= load_melihs_personal_data()


    # TASK 1

    #Computes P(5⋆|Artist = A), P(5⋆|Year = Y), P(5⋆|Explicit = E).

    # some conditional probabilities with one feature
    print("P(5★ | Artist = Doja Cat):",
          p_5_given(global_df, "primary_artist_name", "Doja Cat"))

    print("P(5★ | Year = 2016):",
          p_5_given(global_df, "album_release_year", 2016))
    
    print("P(5★ | danceability: danceable):",
          p_5_given(global_df, "ab_danceability_value", "danceable"))
    
    print("P(5★ | genre = pop):",
          p_5_given(global_df, "ab_genre_rosamerica_value", "pop"))

    print("P(5★ | Explicit = True):",
          p_5_given(global_df, "explicit", True))
    

    # global top artists
    print("\nTop Artists by P(5★):")
    print(top_p5_by_feature(global_df, "primary_artist_name", min_count=20))

    #global top release years
    print("\nTop Release Years by P(5★):")
    print(top_p5_by_feature(global_df, "album_release_year", min_count=50))
    
    

    # TASK 2

    # some conditional probabilities with two features

    # conditional prob with one feature to see the effect of two features
    # Ed Sheeran vs pop & Ed Sheeran
    print("P(5★ | Artist = Ed Sheeran):",
          p_5_given(global_df, "primary_artist_name", "Ed Sheeran"))
    print("P(5★ | Artist = Ed Sheeran & genre = pop):",
          p_5_given_two(global_df,"primary_artist_name","Ed Sheeran","ab_genre_rosamerica_value", "pop"))
    
    # pop vs pop & danceable
    print("P(5★ | genre = pop):",
          p_5_given(global_df, "ab_genre_rosamerica_value", "pop"))
    print("P(5★ | danceability: danceable & genre = pop):",
          p_5_given_two(global_df,"ab_danceability_value","danceable","ab_genre_rosamerica_value", "pop"))
    
    # happy & bright
    print("P(5★ | happy & bright):",
      p_5_given_two(global_df, "ab_mood_happy_value", "happy","ab_timbre_value", "bright"))
    
    # relaxed & acoustic
    print("P(5★ | relaxed & acoustic):",
          p_5_given_two(global_df,"ab_mood_relaxed_value", "relaxed","ab_mood_acoustic_value", "acoustic"))
    
    

    # TASK 3 

    #bayes rule implementation
    print("P(Artist = Halsey | 5★):",
      bayes(global_df, "primary_artist_name", "Halsey"))

    print("P(Artist = Taylor Swift | 5★):",
      bayes(global_df, "primary_artist_name", "Taylor Swift"))
    



    # TASK 4

    # GULCE'S PROB
    print("\nGulce's Top Artists by P(5★):")
    print(top_p5_by_feature(gulce_df, "primary_artist_name", min_count=1))

    print("\n Gulce's Top Release Years by P(5★):")
    print(top_p5_by_feature(gulce_df, "album_release_year", min_count=2))

    print("\nGulce's  Top Genres:")
    print(top_p5_by_feature(gulce_df, "ab_genre_rosamerica_value", min_count=2))

    # MELIH'S PROB
    print("\nMelih's Top Artists by P(5★):")
    print(top_p5_by_feature(melih_df, "primary_artist_name", min_count=1))

    print("\n Melih's Top Release Years by P(5★):")
    print(top_p5_by_feature(melih_df, "album_release_year", min_count=2))

    print("\nMelih's  Top Genres:")
    print(top_p5_by_feature(melih_df, "ab_genre_rosamerica_value", min_count=2))


    # GROUP PROB = total prob / 2 for the same feature
    p_gulce = p_5_given(gulce_df, "ab_genre_rosamerica_value", "pop")
    print("Gulce's P(5★ | genre = pop):", p_5_given(gulce_df, "ab_genre_rosamerica_value", "pop"))
    p_melih = p_5_given(melih_df, "ab_genre_rosamerica_value", "pop")
    print("Melih's P(5★ | genre = pop):", p_5_given(melih_df, "ab_genre_rosamerica_value", "pop"))

    p_group = np.mean([p_gulce, p_melih])
    print("P_group(5★ | genre = pop):", p_group)



    # MOST INFLUENTIAL AFFECTS ON RATINGS
    # finds the most influential ones by best p5 -global 5
    global_p5 = (global_df["rating"] == 5).mean()
    print("Global P(5★):", global_p5)
    
    features = [
    ("ab_danceability_value"),
    ("ab_mood_acoustic_value"),
    ("ab_mood_aggressive_value"),
    ("ab_mood_electronic_value"),
    ("ab_mood_happy_value"),
    ("ab_mood_party_value"),
    ("ab_mood_relaxed_value"),
    ("ab_mood_sad_value"),
    ("ab_gender_value"),
    ("ab_voice_instrumental_value"),
    ("ab_timbre_value"),
    ("ab_genre_dortmund_value"),
    ("ab_genre_rosamerica_value"),
    ]

    print("\nFeature Influence (Lift over Feature's Own Mean):")
    print("-" * 90)
    print(f"{'Feature':30s} | {'Be st Category':15s} | {'Category P(5★)':9s} | {'Feature Mean':9s} | {'Lift'}")
    print("-" * 90)

    for f in features:
        # 1. Get stats for all categories in this feature
        df_feature = top_p5_by_feature(global_df, f, min_count=30, top_n=100)

        if df_feature.empty:
            continue

        # 2. Calculate the weighted average P(5★) for this feature specifically
        # We use (P * count) to get the true mean for that feature
        feature_mean = (df_feature["P(5★)"] * df_feature["count"]).sum() / df_feature["count"].sum()

        # 3. Get the best performing category
        best_row = df_feature.iloc[0]
        best_value = best_row[f]
        best_p5 = best_row["P(5★)"]
        
        # 4. Calculate Lift relative to the feature's own average
        relative_lift = best_p5 - feature_mean

        print(
            f"{f:30s} | "
            f"{str(best_value):15s} | "
            f"{best_p5:.3f}     | "
            f"{feature_mean:.3f}     | "
            f"{relative_lift:+.3f}"
        )
