# Part 2: User Variability Modeling

"""
Part 2: User Variability Modeling

Model how many recommendations it takes for users to rate a song 5★ using 
geometric and Beta-geometric distributions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import beta as beta_func


# compute how many ratings until 5* one and put them in an array
def compute_Tu(ratings):
    """
    Compute T_u for each user: 
    the round index of the first 5★ rating
    """
    Tu = []

    for user, df_u in ratings.groupby("user_id"):
        five_star_rounds = df_u[df_u["rating"] == 5]["round_idx"]

        if len(five_star_rounds) > 0:
            Tu.append(five_star_rounds.min())

    return np.array(Tu)

# estimation of general p for geometric distribution
def fit_geometric(Tu):
    """
    Estimate p for geometric distribution
    """
    mean_T = np.mean(Tu)
    p_hat = 1 / mean_T
    return p_hat

# geometric distribution
def geometric_pmf(t, p):
    return (1 - p)**(t - 1) * p

# beta-geometric distribution
def beta_geometric_pmf(t, alpha, beta):
    return beta_func(alpha + 1, beta + t - 1) / beta_func(alpha, beta)




def main():

    ratings = pd.read_csv("../data/ratings.csv")


    Tu = compute_Tu(ratings)
    p_hat = fit_geometric(Tu)


    print("- Calculate time-to-5★ for each user")
    print(Tu)


    # empirical probability = T = n, (how many users found 5* at nth count) / total user count
    unique, counts = np.unique(Tu, return_counts=True)
    empirical_pmf = counts / counts.sum()

    print("Empirical distribution:")
    for t, p in zip(unique, empirical_pmf):
        print(f"T = {t}: {p:.3f}")


    print("- Fit geometric distribution")
    print("Geometric model prediction:")
    for t in unique:
        print(f"T = {t}: {geometric_pmf(t, p_hat):.3f}")


    # paramaters for a and b
    params = [
    (1, 5),   # pickier users
    (3, 5),   # a bit picky
    (2, 2),   # balanced
    (5, 1)    # users finding a favourite quickly
    ]


    print("- Fit Beta-geometric distribution")
    print("Beta-Geometric model prediction:")
    for alpha, beta in params:
        print(f"\nBeta-Geometric (α={alpha}, β={beta}):")
        for t in unique:
            print(f"T={t}: {beta_geometric_pmf(t, alpha, beta):.3f}")



    print("- Perform hypothesis testing between user groups")
    #mergeing the tracks with ratings to correctly group them
    ratings = pd.read_csv("../data/ratings.csv")
    tracks = pd.read_csv("../data/tracks.csv")
    df = ratings.merge(tracks, left_on="song_id", right_on="track_id")

    first_fives = (
    df[df["rating"] == 5]
    .sort_values("round_idx")
    .groupby("user_id")
    .first()
    .reset_index()
    )

    # hypothesis test due to album release year
    group_old = first_fives[first_fives["album_release_year"] < 2000]
    group_new = first_fives[first_fives["album_release_year"] >= 2000]

    Tu_old = group_old["round_idx"].values
    Tu_new = group_new["round_idx"].values

    print("users who likes songs released before year 2000", Tu_old)
    print("users who likes songs released after year 2000",Tu_new)

    stat, p_value1 = stats.mannwhitneyu(Tu_old, Tu_new, alternative="two-sided")
    print("p-value by release year:", p_value1)

    print("Mean T (old):", Tu_old.mean())
    print("Mean T (new):", Tu_new.mean())

    # hypothesis test due to popularity
    group_notpopular= first_fives[first_fives["track_popularity"] < 80]
    group_popular = first_fives[first_fives["track_popularity"] >= 80]

    Tu_notpopular = group_notpopular["round_idx"].values
    Tu_popular= group_popular["round_idx"].values

    print("users who likes songs that are not popular", Tu_notpopular)
    print("users who likes songs that are popular",Tu_popular)

    stat, p_value2 = stats.mannwhitneyu(Tu_notpopular, Tu_popular, alternative="two-sided")
    print("p-value by popularity:", p_value2)

    print("Mean T not popular:", Tu_notpopular.mean())
    print("Mean T popular:", Tu_popular.mean())



if __name__ == "__main__":
    main()
