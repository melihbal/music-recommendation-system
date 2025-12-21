import pandas as pd
import numpy as np
import math


# ------------------------------------------
# Math and Probability
# ------------------------------------------
#Log Beta function: ln(Beta(a, b))
def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# PMF for geometric distribution
def geometric_pmf(t, p):
    return (1 - p) ** (t - 1) * p

# PMF for Beta-Geometric Distribution.
# P(T=t) = Beta(alpha+1, beta+t-1) / Beta(alpha, beta)
def beta_geometric_pmf(t, alpha, beta_param):
    # Calculation in log-space to ensure numerical stability
    log_val = math.lgamma(alpha + 1) + math.lgamma(beta_param + t - 1) - math.lgamma(alpha + beta_param + t)
    log_val -= lbeta(alpha, beta_param)
    return math.exp(log_val)

# Returns U statistic and p-value
def mann_whitney_u_test(x, y):
    n1 = len(x)
    n2 = len(y)

    if n1 == 0 or n2 == 0:
        return 0, 1.0

    # Combine data and compute ranks (using pandas for average rank handling)
    # 0 for group x, 1 for group y
    df = pd.DataFrame({'val': np.concatenate([x, y]),
                       'group': [0] * n1 + [1] * n2})

    # Compute ranks
    df['rank'] = df['val'].rank(method='average')

    # Sum of ranks for first group
    r1 = df[df['group'] == 0]['rank'].sum()

    # U statistic formula
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation for p-value
    mu_u = n1 * n2 / 2

    # Standard deviation
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    if sigma_u == 0:
        z = 0
    else:
        z = (u - mu_u) / sigma_u

    # Two-sided p-value using error funtction
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return u, p_val


# ---------------------------------------------
# Data Processing & Modeling Functions
# ---------------------------------------------


# Computes the index of first 5-star
def compute_Tu(ratings):
    Tu = []
    for user, df_u in ratings.groupby("user_id"):
        # Filter for 5-star ratings
        five_star_rounds = df_u[df_u["rating"] == 5]["round_idx"]

        if len(five_star_rounds) > 0:
            # +1 because round_idx is 0-indexed
            Tu.append(five_star_rounds.min() + 1)

    return np.array(Tu)


# E[T] = 1/p  =>  p = 1/mean(T)
def fit_geometric(Tu):
    if len(Tu) == 0:
        return 0.5
    mean_T = np.mean(Tu)
    return 1.0 / mean_T



#Estimate alpha and beta for Beta geometric distribution using grid search maximum likelihood estimation (MLE).
def fit_beta_geometric_mle(Tu):
    if len(Tu) == 0:
        return 1.0, 1.0

    unique_t, counts = np.unique(Tu, return_counts=True)

    best_alpha = 1.0
    best_beta = 1.0
    best_ll = -np.inf

    # Define search grid
    alpha_range = np.linspace(0.1, 10.0, 25)
    beta_range = np.linspace(0.1, 25.0, 25)

    for a in alpha_range:
        for b in beta_range:
            log_likelihood = 0
            for t, count in zip(unique_t, counts):
                prob = beta_geometric_pmf(t, a, b)
                if prob > 0:
                    log_likelihood += count * math.log(prob)
                else:
                    log_likelihood += count * -1e9  # Penalty for 0 prob

            if log_likelihood > best_ll:
                best_ll = log_likelihood
                best_alpha = a
                best_beta = b

    return best_alpha, best_beta


# ==========================================
# Main Execution
# ==========================================

def main():
    print("--- Part 2: User Variability Modeling ---\n")

    # Load Ratings
    try:
        ratings = pd.read_csv("../data/user_ratings.csv")
    except FileNotFoundError:
        try:
            # Fallback for local testing
            ratings = pd.read_csv("data/user_ratings.csv")
        except FileNotFoundError:
            print("Error: 'user_ratings.csv' not found in ../data/ or data/")
            return

    # Compute Tu and global models
    Tu = compute_Tu(ratings)
    print(f"Data Loaded: Found {len(Tu)} users who gave a 5* rating.")

    # geometric fit
    p_hat = fit_geometric(Tu)
    print(f"Geometric Model: Estimated p = {p_hat:.4f}")

    # Beta-geometric fit
    print("Fitting Beta-Geometric Model (Grid Search MLE)...")
    alpha_hat, beta_hat = fit_beta_geometric_mle(Tu)
    print(f"Beta-Geometric Model: Estimated alpha = {alpha_hat:.2f}, beta = {beta_hat:.2f}")

    # 3. Build comparison table (Cumulative distribution)
    max_T = Tu.max()
    display_limit = min(20, max_T)
    all_t = np.arange(1, display_limit + 1)

    # Empirical CDF
    unique, counts = np.unique(Tu, return_counts=True)
    empirical_pmf = pd.Series(counts, index=unique).reindex(all_t, fill_value=0) / len(Tu)
    empirical_cdf = empirical_pmf.cumsum()

    # Geometric CDF
    geom_cdf = [1 - (1 - p_hat) ** t for t in all_t]

    # Beta-geometric CDF
    bg_pmf = [beta_geometric_pmf(t, alpha_hat, beta_hat) for t in all_t]
    bg_cdf = np.cumsum(bg_pmf)

    # Construct DataFrame
    df_cdf = pd.DataFrame({
        "Round (T)": all_t,
        "Empirical CDF": empirical_cdf,
        f"Geo(p={p_hat:.2f})": geom_cdf,
        f"BetaGeo(a={alpha_hat:.1f},b={beta_hat:.1f})": bg_cdf
    })

    print("\n--- Model Comparison (Cumulative Probability) ---")
    print(df_cdf.set_index("Round (T)").round(3))

    # 4. Hypothesis testing
    print("\n" + "=" * 40)
    print("HYPOTHESIS TESTING (Manual Mann-Whitney U)")
    print("=" * 40)

    try:
        tracks = pd.read_csv("../data/tracks.csv")
    except FileNotFoundError:
        try:
            tracks = pd.read_csv("data/tracks.csv")
        except FileNotFoundError:
            print("Error: 'tracks.csv' not found in ../data/ or data/")
            return

    # Filter only the successful 5* rating events
    success_events = ratings[ratings['rating'] == 5].copy()

    # Get the first 5-star rating for each user
    success_events = success_events.sort_values('round_idx')
    first_success = success_events.groupby('user_id').first().reset_index()

    # Merge with tracks
    merged = first_success.merge(tracks, left_on='song_id', right_on='track_id')

    # Define T_u for these rows
    merged['Tu'] = merged['round_idx'] + 1

    # --- Test 1: Release Year ---
    group_old = merged[merged["album_release_year"] < 2010]['Tu'].values
    group_new = merged[merged["album_release_year"] >= 2010]['Tu'].values

    u1, p1 = mann_whitney_u_test(group_old, group_new)
    print(f"\n[Test 1] Release Year (<2010 vs >=2010)")
    print(f"  Mean Wait (Old): {np.mean(group_old):.2f} (n={len(group_old)})")
    print(f"  Mean Wait (New): {np.mean(group_new):.2f} (n={len(group_new)})")
    print(f"  p-value: {p1:.4f} {'Significant' if p1 < 0.05 else 'Insignificant'}")

    # --- Test 2: Popularity ---
    group_niche = merged[merged["track_popularity"] < 70]['Tu'].values
    group_pop = merged[merged["track_popularity"] >= 70]['Tu'].values

    u2, p2 = mann_whitney_u_test(group_niche, group_pop)
    print(f"\n[Test 2] Popularity (<70 vs >=70)")
    print(f"  Mean Wait (Niche): {np.mean(group_niche):.2f}")
    print(f"  Mean Wait (Pop):   {np.mean(group_pop):.2f}")
    print(f"  p-value: {p2:.4f} {'Significant' if p2 < 0.05 else 'Insignificant'}")

    # --- Test 3: Mood (Party) ---
    group_party = merged[merged["ab_mood_party_value"] == "party"]['Tu'].values
    group_chill = merged[merged["ab_mood_party_value"] == "not_party"]['Tu'].values

    u3, p3 = mann_whitney_u_test(group_party, group_chill)
    print(f"\n[Test 3] Mood (Party vs Not Party)")
    print(f"  Mean Wait (Party): {np.mean(group_party):.2f}")
    print(f"  Mean Wait (Chill): {np.mean(group_chill):.2f}")
    print(f"  p-value: {p3:.4f} {'Significant' if p3 < 0.05 else 'Insignificant'}")

    # --- Test 4: Duration (Long vs Short) ---
    median_duration = merged["duration_ms"].median()
    group_long = merged[merged["duration_ms"] >= median_duration]["Tu"].values
    group_short = merged[merged["duration_ms"] < median_duration]["Tu"].values

    u4, p4 = mann_whitney_u_test(group_long, group_short)
    print(f"\n[Test 4] Duration (Long vs Short, median={median_duration / 1000:.1f}s)")
    print(f"  Mean Wait (Long):  {np.mean(group_long):.2f} (n={len(group_long)})")
    print(f"  Mean Wait (Short): {np.mean(group_short):.2f} (n={len(group_short)})")
    print(f"  p-value: {p4:.4f} {'Significant' if p4 < 0.05 else 'Insignificant'}")

    # --- Test 4: Popularity Extremes (Bottom 25% vs Top 25%) ---
    # Finding: Extremely popular songs found much faster (p ~ 0.044)
    pop_low = merged['track_popularity'].quantile(0.25)
    pop_high = merged['track_popularity'].quantile(0.75)

    group_bottom25 = merged[merged["track_popularity"] <= pop_low]['Tu'].values
    group_top25 = merged[merged["track_popularity"] >= pop_high]['Tu'].values

    u4, p4 = mann_whitney_u_test(group_bottom25, group_top25)
    print(f"\n[Test 5] Popularity Extremes (Bottom 25% vs Top 25%)")
    print(f"  Mean Wait (Bottom 25%): {np.mean(group_bottom25):.2f} (n={len(group_bottom25)})")
    print(f"  Mean Wait (Top 25%):    {np.mean(group_top25):.2f} (n={len(group_top25)})")
    print(f"  p-value: {p4:.4f} {'Significant' if p4 < 0.05 else 'Insignificant'}")


if __name__ == "__main__":
    main()