
# Part 4: Monte Carlo Evaluation

"""
Part 4: Monte Carlo Evaluation

Evaluate recommendation models using statistical simulation and Monte Carlo methods.
"""
import time

import pandas as pd
import numpy as np
from part3 import recommend_safe, recommend_probabilistic


def main():
    print("Part 4: Monte Carlo Evaluation")
    print("TODO: Implement Monte Carlo simulation")
    print("- Simulate user interactions")

    print("- Compute Hit@k, Average Rating, Time-to-5★")



def hit_at_k(recommender_fn, user_histories, k):
    hits = 0
    total = len(user_histories)

    for history, future_5star_ids in user_histories:
        recs = recommender_fn(history, k)
        rec_ids = {r[0] for r in recs}

        if rec_ids & future_5star_ids:
            hits += 1

    return hits / total if total > 0 else 0



def build_user_histories(df, history_len=10):
    user_histories = []

    for user_id, u_df in df.groupby('user_id'):
        u_df = u_df.sort_values('round_idx')

        if len(u_df) <= history_len:
            continue

        history = list(
            u_df[['song_id', 'track_name', 'rating']]
            .head(history_len)
            .itertuples(index=False, name=None)
        )

        # Create a slice for the future part of the dataframe
        future_df = u_df.iloc[history_len:]

        # Filter that slice for 5-star ratings
        future_5star_ids = set(future_df[future_df['rating'] == 5]['song_id'])

        if future_5star_ids:
            user_histories.append((history, future_5star_ids))

    return user_histories



def average_rating_at_k(recommender_fn, df, history_len=10, k=10):
    all_ratings = []

    for user_id, u_df in df.groupby('user_id'):
        u_df = u_df.sort_values('round_idx')

        if len(u_df) <= history_len:
            continue

        history = list(
            u_df[['song_id', 'track_name', 'rating']]
            .head(history_len)
            .itertuples(index=False, name=None)
        )

        # Run recommender
        recs = recommender_fn(history, k)
        rec_ids = {r[0] for r in recs}

        # Get ratings for recommended songs that the user actually rated
        rated_recs = u_df[u_df['song_id'].isin(rec_ids)]

        all_ratings.extend(rated_recs['rating'].tolist())

    return sum(all_ratings) / len(all_ratings) if all_ratings else 0



def compute_Tu(recommender_fn, df, history_len=10, k=10, max_rounds=20):
    Tu = []

    for user_id, u_df in df.groupby("user_id"):
        u_df = u_df.sort_values("round_idx").reset_index(drop=True)

        if len(u_df) <= history_len:
            continue

        # Initial history
        history = list(
            u_df.loc[:history_len-1, ['song_id', 'track_name', 'rating']]
            .itertuples(index=False, name=None)
        )

        # Future interactions (RESET INDEX!)
        future = u_df.loc[history_len:].reset_index(drop=True)

        found = False
        round_num = 0

        for i in range(min(len(future), max_rounds)):
            round_num += 1

            # Model recommendation
            recs = recommender_fn(history, k)
            rec_ids = {r[0] for r in recs}

            # 5★ songs revealed up to this point
            available_5stars = set(
                future.iloc[:i+1][future.iloc[:i+1]["rating"] == 5]["song_id"]
            )

            if rec_ids & available_5stars:
                Tu.append(round_num)
                found = True
                break

            # Offline simulation: user gives next rating
            history.append(
                tuple(future.iloc[i][['song_id', 'track_name', 'rating']])
            )

        if not found:
            Tu.append(max_rounds + 1)  # censored

    return np.array(Tu)



    print("- Calculate confidence intervals")
    print("- Compare model performance")

if __name__ == "__main__":

    rounds = 20

    for i in range(rounds):
        # Read ratings
        df = pd.read_csv('../data/user_ratings.csv')

        # Build evaluation data
        user_histories = build_user_histories(df, history_len=5)

        k = 10
        random_seed = int(time.time())
        np.random.seed(random_seed)  # for reproducibility


        hit_safe = hit_at_k(recommend_safe,user_histories, k)
        hit_prob = hit_at_k(recommend_probabilistic,user_histories,k)

        avg_rate_safe= average_rating_at_k(recommend_safe, df, history_len=10, k=10)
        avg_rate_prob= average_rating_at_k(recommend_probabilistic, df, history_len=10, k=10)

        Tu_safe= compute_Tu(recommend_safe, df, history_len=10, k=10,max_rounds=20)
        Tu_prob= compute_Tu(recommend_probabilistic, df, history_len=10, k=10,max_rounds=20)


        print(f"Hit@{k} (Popularity-Biased / Safe): {hit_safe:.3f}")
        print(f"Hit@{k} (Utility-Based / Probabilistic): {hit_prob:.3f}")

        print(f"average (Popularity-Biased / Safe): {avg_rate_safe:.3f}")
        print(f"average (Utility-Based / Probabilistic): {avg_rate_prob:.3f}")

        print(f"Mean Tu (Popularity-Biased / Safe): {Tu_safe.mean():.3f}")
        print(f"Median Tu (Popularity-Biased / Safe): {np.median(Tu_safe):.3f}")
        print(f"Mean Tu (Utility-Based / Probabilistic): {Tu_prob.mean():.3f}")
        print(f"Median Tu (Utility-Based / Probabilistic): {np.median(Tu_prob):.3f}")




