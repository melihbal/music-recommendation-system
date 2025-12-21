
# Part 4: Monte Carlo Evaluation

"""
Part 4: Monte Carlo Evaluation

Evaluate recommendation models using statistical simulation and Monte Carlo methods.
"""
import time

import pandas as pd
import numpy as np
from part3 import recommend_safe, recommend_probabilistic



# Simulates user sessions by splitting historical data into a seed history and a subsequent 'future' set for evaluation.
# returns (history, future) tuple for every user
def build_user_histories_random(df, history_len=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    user_histories = []

    # Iterate through each user's rating history
    for user_id, u_df in df.groupby("user_id"):

        # Ensure ratings are in chronological order
        u_df = u_df.sort_values("round_idx").reset_index(drop=True)

        # Skip users who don't have enough interactions to form a history + future
        if len(u_df) <= history_len:
            continue

        # Calculate the latest possible starting point for the history window
        max_start = len(u_df) - history_len - 1
        if max_start <= 0:
            continue

        # Randomly select a starting point to increase simulation variety
        start_idx = np.random.randint(0, max_start)

        history = list(
            u_df.loc[start_idx:start_idx + history_len - 1,
                     ["song_id", "track_name", "rating"]]
            .itertuples(index=False, name=None)
        )

        # Extract everything after the history: used to 'ground truth' the model's performance
        future_df = u_df.loc[start_idx + history_len:].reset_index(drop=True)

        user_histories.append((history, future_df))

    return user_histories



# Calculates the Hit@k metric: the proportion of users who were recommended at least one song they later rated as 5 stars in k try.
# returns the probability
def hit_at_k(recommender_fn, user_histories, k):
    hits = 0
    total = 0

    for history, future_df in user_histories:

        # Identify all songs in the test set (future) that the user gave 5 stars.
        future_5star_ids = set(
            future_df[future_df["rating"] == 5]["song_id"]
        )

        # If the user has no 5-star songs in the future, we skip them as a 'hit' is impossible to achieve for this simulation trial.
        if not future_5star_ids:
            continue

        # Increment valid trials count for users who have 5-star ground truth available
        total +=1 

        # Get the top k recommendations from the model based on the seed history
        recs = recommender_fn(history, k)
        rec_ids = {r[0] for r in recs}

        # Check if there is any overlap between recommended songs and the user's actual 5-star favorites, if yes: hit
        if rec_ids & future_5star_ids:
            hits += 1

    # Return the hit rate: (Number of users with a hit) / (Total valid users)
    return hits / total if total > 0 else 0



#Calculates the mean rating of recommended songs that the user actually interacted with in the future set. 
# Returns the global average rating across all matched recommendations.
def average_rating_at_k(recommender_fn, user_histories, k):
    ratings = []

    # Get the top k recommendations from the model
    for history, future_df in user_histories:
        recs = recommender_fn(history, k)
        rec_ids = {r[0] for r in recs}

        # Filter the user's future interactions to find songs that were also recommended
        matched = future_df[future_df["song_id"].isin(rec_ids)]

        # Collect the actual ratings given by the user to these recommended songs
        ratings.extend(matched["rating"].tolist())

    # Return the global average rating across all matched recommendations.
    return np.mean(ratings) if ratings else 0



#Simulates a sequence of recommendation rounds (time) to find the first 5-star hit Tu measures the speed of the recommendation system.
# returns array of users round count till the first 5 star.
def compute_Tu(recommender_fn, user_histories, k=10, max_rounds=30):
    Tu = []

    for history, future_df in user_histories:
        hist = history.copy()
        found = False

        # Pre-identify which songs in the future would satisfy the user (5 stars)
        future_5star_ids = set(
            future_df[future_df["rating"] == 5]["song_id"]
        )

        # Handle users with no 5-star songs, penalize the model with (max_rounds + 1)
        if not future_5star_ids:
            Tu.append(max_rounds + 1)
            continue

        # model generates recommendations based on current history
        # Simulate round-by-round interaction
        for t in range(max_rounds):

            # Generate recommendations based on the updated history
            recs = recommender_fn(hist, k)
            rec_ids = {r[0] for r in recs}

            # If success: record the round index and exit the loop
            if rec_ids & future_5star_ids:
                Tu.append(t + 1)
                found = True
                break

            # If failure: update the history with the next real interaction to simulate the model learning from user feedback
            if t < len(future_df):
                hist.append(tuple(
                    future_df.iloc[t][["song_id", "track_name", "rating"]]
                ))

        if not found:
            # If 5-star song is never found within the time limit
            Tu.append(max_rounds + 1)

    return np.array(Tu)



#Calculates the 95% Confidence Interval (CI) for the mean difference between Model A and Model B.
def mean_ci(diff):

    diff = np.array(diff)
    n = len(diff)

    # Calculate the sample mean of the differences
    mean = diff.mean()

    # Calculate the sample variance and standard deviation
    variance = np.sum((diff - mean)**2) / (n - 1)
    std = np.sqrt(variance)

    # 1.96 is the critical value for a two-tailed normal distribution
    z = 1.96  # 95% confidence

    # Calculate the lower and upper bounds of the confidence interval with standast error (std / sqrt(n)).
    ci_low  = mean - z * std / np.sqrt(n)
    ci_high = mean + z * std / np.sqrt(n)

    return mean, ci_low, ci_high




if __name__ == "__main__":

    # Read ratings
    df = pd.read_csv('data/user_ratings.csv')

    # Initialize lists to store metrics for each Monte Carlo trial
    hit_safe_list = []
    hit_prob_list = []

    avg_safe_list = []
    avg_prob_list = []

    Tu_safe_list = []
    Tu_prob_list = []

    # Simulation parameters
    rounds = 1000 # Number of Monte Carlo iterations for statistical significance
    k = 10 # Recommendation list size


    print(f"Starting Monte Carlo Evaluation for {rounds} rounds:")

    for i in range(rounds):

        # Generate random history/future splits for each user in this trial
        user_histories = build_user_histories_random(df, history_len=10, seed=None)

        # Evaluate Hit@k
        hit_safe_list.append(hit_at_k(recommend_safe, user_histories, k))
        hit_prob_list.append(hit_at_k(recommend_probabilistic, user_histories, k))

        # Evaluate Average Rating
        avg_safe_list.append(average_rating_at_k(recommend_safe, user_histories, k))
        avg_prob_list.append(average_rating_at_k(recommend_probabilistic, user_histories, k))

        # Evaluate Time-to-5★ (Tu)
        Tu_safe_list.append(compute_Tu(recommend_safe, user_histories, k, max_rounds=30).mean())
        Tu_prob_list.append(compute_Tu(recommend_probabilistic, user_histories, k, max_rounds=30).mean())

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{rounds} rounds completed")
    


    # array of differences between all elements in the lists
    diff_hit = np.array(hit_safe_list) - np.array(hit_prob_list)
    diff_avg = np.array(avg_safe_list) - np.array(avg_prob_list)
    diff_Tu  = np.array(Tu_safe_list) - np.array(Tu_prob_list) 

    # Compute 95% Confidence Intervals for the differences
    hit_mean, hit_ci_low, hit_ci_high = mean_ci(diff_hit)
    avg_mean, avg_ci_low, avg_ci_high = mean_ci(diff_avg)
    Tu_mean,  Tu_ci_low,  Tu_ci_high  = mean_ci(diff_Tu)



    print("\n=== MONTE CARLO RESULTS ===")

    print(f"Hit@{k} Safe: {np.mean(hit_safe_list):.3f}")
    print(f"Hit@{k} Prob: {np.mean(hit_prob_list):.3f}")

    print(f"Avg Rating Safe: {np.mean(avg_safe_list):.3f}")
    print(f"Avg Rating Prob: {np.mean(avg_prob_list):.3f}")

    print(f"Mean Tu Safe: {np.mean(Tu_safe_list):.3f}")
    print(f"Mean Tu Prob: {np.mean(Tu_prob_list):.3f}")



    print("\n=== 95% CONFIDENCE INTERVALS ===")

    print(f"Hit@{k} Difference (Safe - Prob): "
        f"{hit_mean:.3f} [{hit_ci_low:.3f}, {hit_ci_high:.3f}]")

    print(f"Average Rating Difference (Safe - Prob): "
        f"{avg_mean:.3f} [{avg_ci_low:.3f}, {avg_ci_high:.3f}]")

    print(f"Time-to-5★ Difference (Safe - Prob): "
        f"{Tu_mean:.3f} [{Tu_ci_low:.3f}, {Tu_ci_high:.3f}]")




