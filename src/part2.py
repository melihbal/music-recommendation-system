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
            Tu.append(five_star_rounds.min() + 1)

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
    # 1. Load Data
    try:
        ratings = pd.read_csv("../data/user_ratings.csv")
    except FileNotFoundError:
        print("Error: user_ratings.csv not found in ../data/")
        return

    # 2. Global Modeling (Tu and p_hat)
    # Ensure your compute_Tu function has the '+ 1' fix!
    Tu = compute_Tu(ratings)
    p_hat = fit_geometric(Tu)

    print(f"Global Statistics:")
    print(f"- Analyzed {len(Tu)} users")
    print(f"- Estimated Geometric p: {p_hat:.4f}\n")

    # 3. Build Model Comparison Table (Cumulative / CDF)
    max_T = Tu.max()
    all_t = np.arange(1, max_T + 1)

    # Calculate Empirical PMF and reindex to full range (filling gaps with 0)
    unique, counts = np.unique(Tu, return_counts=True)
    empirical_series = pd.Series(counts / counts.sum(), index=unique)
    empirical_series = empirical_series.reindex(all_t, fill_value=0)

    # Step A: Build PMF DataFrame on the FULL range 'all_t'
    pmf_df = pd.DataFrame(index=all_t)
    pmf_df["Empirical"] = empirical_series
    pmf_df[f"Geom(p={p_hat:.2f})"] = [geometric_pmf(t, p_hat) for t in all_t]

    # Step B: Add Beta-Geometric PMFs
    params = [
        (1, 5),  # pickier
        (3, 5),  # moderate
        (2, 2),  # balanced
        (5, 1)  # easy
    ]

    for alpha, beta in params:
        col_name = f"BG({alpha},{beta})"
        pmf_df[col_name] = [beta_geometric_pmf(t, alpha, beta) for t in all_t]

    # Step C: Convert PMF to Cumulative Distribution (CDF)
    # This answers: "What % of users found a song BY round T?"
    cdf_df = pmf_df.cumsum()
    cdf_df.index.name = "T"

    # Print Table
    print("--- Cumulative Distribution ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(cdf_df.round(3))
    print("\n")


    # 4. Hypothesis Testing
    print("-"*50)
    print("HYPOTHESIS TESTING")
    print("-"*50)

    try:
        tracks = pd.read_csv("../data/tracks.csv")
    except FileNotFoundError:
        print("Error: tracks.csv not found.")
        return

    # Merge to get metadata for every rating
    df = ratings.merge(tracks, left_on="song_id", right_on="track_id")

    # Find the FIRST 5-star rating for each user
    first_fives = (
        df[df["rating"] == 5]
        .sort_values("round_idx")
        .groupby("user_id")
        .first()
        .reset_index()
    )

    # CRITICAL: Create 1-based 'Tu' for testing
    first_fives["Tu"] = first_fives["round_idx"] + 1

    # --- Test 1: Album Release Year (<2000 vs >=2000) ---
    group_old = first_fives[first_fives["album_release_year"] < 2000]["Tu"]
    group_new = first_fives[first_fives["album_release_year"] >= 2000]["Tu"]

    stat1, p1 = stats.mannwhitneyu(group_old, group_new, alternative="two-sided")
    print(f"\n[Test 1] Release Year (<2000 vs >=2000)")
    print(f"  Mean Wait (Old): {group_old.mean():.2f}")
    print(f"  Mean Wait (New): {group_new.mean():.2f}")
    print(f"  p-value: {p1:.4f} ({'Significant' if p1 < 0.05 else 'Not Significant'})")

    # --- Test 2: Popularity (<80 vs >=80) ---
    group_niche = first_fives[first_fives["track_popularity"] < 80]["Tu"]
    group_pop = first_fives[first_fives["track_popularity"] >= 80]["Tu"]

    stat2, p2 = stats.mannwhitneyu(group_niche, group_pop, alternative="two-sided")
    print(f"\n[Test 2] Popularity (<80 vs >=80)")
    print(f"  Mean Wait (Niche): {group_niche.mean():.2f}")
    print(f"  Mean Wait (Pop):   {group_pop.mean():.2f}")
    print(f"  p-value: {p2:.4f} ({'Significant' if p2 < 0.05 else 'Not Significant'})")

    # --- Test 3: Mood (Party vs Not Party) ---
    group_party = first_fives[first_fives["ab_mood_party_value"] == "party"]["Tu"]
    group_chill = first_fives[first_fives["ab_mood_party_value"] == "not_party"]["Tu"]

    stat3, p3 = stats.mannwhitneyu(group_party, group_chill, alternative="two-sided")
    print(f"\n[Test 3] Mood (Party vs Not Party)")
    print(f"  Mean Wait (Party): {group_party.mean():.2f}")
    print(f"  Mean Wait (Chill): {group_chill.mean():.2f}")
    print(f"  p-value: {p3:.4f} ({'Significant' if p3 < 0.05 else 'Not Significant'})")

    # --- Test 4: Duration (Split by Median) ---
    median_duration = first_fives["duration_ms"].median()
    group_long = first_fives[first_fives["duration_ms"] >= median_duration]["Tu"]
    group_short = first_fives[first_fives["duration_ms"] < median_duration]["Tu"]

    stat4, p4 = stats.mannwhitneyu(group_long, group_short, alternative="two-sided")
    print(f"\n[Test 4] Duration (Long vs Short, median={median_duration/1000:.1f}s)")
    print(f"  Mean Wait (Long):  {group_long.mean():.2f}")
    print(f"  Mean Wait (Short): {group_short.mean():.2f}")
    print(f"  p-value: {p4:.4f} ({'Significant' if p4 < 0.05 else 'Not Significant'})")


if __name__ == "__main__":
    main()
