
# Part 4: Monte Carlo Evaluation

"""
Part 4: Monte Carlo Evaluation

Evaluate recommendation models using statistical simulation and Monte Carlo methods.
"""
import time

import pandas as pd
import numpy as np
from part3 import recommend_safe, recommend_probabilistic



#function that randomly produces user_histories
def build_user_histories_random(df, history_len=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    user_histories = []

    for user_id, u_df in df.groupby("user_id"):
        u_df = u_df.sort_values("round_idx").reset_index(drop=True)

        if len(u_df) <= history_len:
            continue

        max_start = len(u_df) - history_len - 1
        if max_start <= 0:
            continue

        start_idx = np.random.randint(0, max_start)

        history = list(
            u_df.loc[start_idx:start_idx + history_len - 1,
                     ["song_id", "track_name", "rating"]]
            .itertuples(index=False, name=None)
        )

        future_df = u_df.loc[start_idx + history_len:].reset_index(drop=True)

        user_histories.append((history, future_df))

    return user_histories




def hit_at_k(recommender_fn, user_histories, k):
    hits = 0
    total = len(user_histories)

    for history, future_df in user_histories:
        future_5star_ids = set(
            future_df[future_df["rating"] == 5]["song_id"]
        )

        if not future_5star_ids:
            continue

        recs = recommender_fn(history, k)
        rec_ids = {r[0] for r in recs}

        if rec_ids & future_5star_ids:
            hits += 1

    return hits / total if total > 0 else 0



def average_rating_at_k(recommender_fn, user_histories, k):
    ratings = []

    for history, future_df in user_histories:
        recs = recommender_fn(history, k)
        rec_ids = {r[0] for r in recs}

        matched = future_df[future_df["song_id"].isin(rec_ids)]
        ratings.extend(matched["rating"].tolist())

    return np.mean(ratings) if ratings else 0



def compute_Tu(recommender_fn, user_histories, k=10, max_rounds=30):
    Tu = []

    for history, future_df in user_histories:
        hist = history.copy()
        found = False

        future_5star_ids = set(
            future_df[future_df["rating"] == 5]["song_id"]
        )

        if not future_5star_ids:
            Tu.append(max_rounds + 1)
            continue

        for t in range(max_rounds):
            recs = recommender_fn(hist, k)
            rec_ids = {r[0] for r in recs}

            if rec_ids & future_5star_ids:
                Tu.append(t + 1)
                found = True
                break

            if t < len(future_df):
                hist.append(tuple(
                    future_df.iloc[t][["song_id", "track_name", "rating"]]
                ))

        if not found:
            Tu.append(max_rounds + 1)

    return np.array(Tu)


def mean_ci(diff, alpha=0.05):
    """
    diff: np.array of differences (Model A - Model B)
    returns: mean, (ci_low, ci_high)
    """
    diff = np.array(diff)
    n = len(diff)

    mean = diff.mean()
    variance = np.sum((diff - mean)**2) / (n - 1)
    std = np.sqrt(variance)

    z = 1.96  # 95% confidence
    ci_low  = mean - z * std / np.sqrt(n)
    ci_high = mean + z * std / np.sqrt(n)

    return mean, ci_low, ci_high










if __name__ == "__main__":


    hit_safe_list = []
    hit_prob_list = []

    avg_safe_list = []
    avg_prob_list = []

    Tu_safe_list = []
    Tu_prob_list = []


    rounds = 10
    k = 10

    for i in range(rounds):
        # Read ratings
        df = pd.read_csv('../data/user_ratings.csv')


        user_histories = build_user_histories_random(
        df, history_len=10, seed=i
        )

        hit_safe = hit_at_k(recommend_safe,user_histories, k)
        hit_prob = hit_at_k(recommend_probabilistic,user_histories,k)

        avg_rate_safe= average_rating_at_k(recommend_safe, user_histories, k=10)
        avg_rate_prob= average_rating_at_k(recommend_probabilistic, user_histories, k=10)

        Tu_safe= compute_Tu(recommend_safe, user_histories, k=10,max_rounds=30)
        Tu_prob= compute_Tu(recommend_probabilistic, user_histories, k=10,max_rounds=30)


        print(f"Hit@{k} (Popularity-Biased / Safe): {hit_safe:.3f}")
        print(f"Hit@{k} (Utility-Based / Probabilistic): {hit_prob:.3f}")

        print(f"average (Popularity-Biased / Safe): {avg_rate_safe:.3f}")
        print(f"average (Utility-Based / Probabilistic): {avg_rate_prob:.3f}")

        print(f"Mean Tu (Popularity-Biased / Safe): {Tu_safe.mean():.3f}")
        print(f"Median Tu (Popularity-Biased / Safe): {np.median(Tu_safe):.3f}")
        print(f"Mean Tu (Utility-Based / Probabilistic): {Tu_prob.mean():.3f}")
        print(f"Median Tu (Utility-Based / Probabilistic): {np.median(Tu_prob):.3f}")


        hit_safe_list.append(hit_safe)
        hit_prob_list.append(hit_prob)

        avg_safe_list.append(avg_rate_safe)
        avg_prob_list.append(avg_rate_prob)

        Tu_safe_list.append(Tu_safe.mean())
        Tu_prob_list.append(Tu_prob.mean())

    
    
    
    # array of differences between all elements in the lists
    diff_hit = np.array(hit_safe_list) - np.array(hit_prob_list)
    diff_avg = np.array(avg_safe_list) - np.array(avg_prob_list)
    diff_Tu  = np.array(Tu_safe_list) - np.array(Tu_prob_list) 

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

    print(f"Time-to-5★ Difference (Prob - Safe): "
        f"{Tu_mean:.3f} [{Tu_ci_low:.3f}, {Tu_ci_high:.3f}]")




