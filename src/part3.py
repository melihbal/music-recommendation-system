# Part 3: Recommender Design

"""
Part 3: Recommender Design

Design and implement two different recommendation algorithms for the music system.
"""

import pandas as pd

print("Initializing Recommender...")

# Constants
MEDIAN_DURATION_MS = 233712  # From Part 2 Findings
ALPHA = 1  # Laplace smoothing factor

try:
    # Load Data
    TRACKS = pd.read_csv("data/tracks.csv")
    RATINGS = pd.read_csv("data/user_ratings.csv")  # Required for Part 1 stats

    # Merge to link ratings to artists/genres
    df_merged = RATINGS.merge(TRACKS, left_on="song_id", right_on="track_id")


    # Precompute probabilities -- Part 1

    # P(5-star | Feature) for every artist and genre
    def compute_conditional_prob(df, feature_col):
        # Group by feature
        groups = df.groupby(feature_col)['rating']

        # Calculate counts: Total ratings and 5-star ratings
        counts = groups.count()
        five_star_counts = groups.apply(lambda x: (x == 5).sum())

        # Laplace smoothing
        probs = (five_star_counts + ALPHA) / (counts + 2 * ALPHA)
        return probs.to_dict()


    print("Computing conditional probabilities (Part 1 logic)...")
    ARTIST_PROBS = compute_conditional_prob(df_merged, 'primary_artist_name')
    GENRE_PROBS = compute_conditional_prob(df_merged, 'ab_genre_rosamerica_value')

    # Global Average P(5-star) for fallback
    GLOBAL_P5 = (len(df_merged[df_merged['rating'] == 5]) + ALPHA) / (len(df_merged) + 2 * ALPHA)

    # Map track popularity for speed
    TRACK_SCORES = TRACKS.set_index('track_id')['track_popularity'].to_dict()

except FileNotFoundError:
    print("WARNING: Could not load data files. Recommender will fail.")
    TRACKS = pd.DataFrame()
    ARTIST_PROBS = {}
    GENRE_PROBS = {}
    GLOBAL_P5 = 0.2

# user history: List[Tuple[str, str, int]]
#                   (track_id, track_name, rating)
# Recommends safe songs deterministically
def recommend_safe(user_history, topk=5):
    liked_ids = set()
    seen_ids = set()

    for track_id, _, rating in user_history:
        if rating > 4:
            liked_ids.add(track_id)
    for x in user_history:
        seen_ids.add(x[0])

    # if no song is liked, return global favourites
    if not liked_ids:
        return recommend_global_hits(seen_ids, topk)

    liked_meta = TRACKS[TRACKS['track_id'].isin(liked_ids)]
    liked_artists = liked_meta['primary_artist_name'].unique()
    liked_genres = liked_meta['ab_genre_rosamerica_value'].unique()

    # Eliminate songs that are already seen
    candidates = TRACKS[~TRACKS['track_id'].isin(seen_ids)].copy()

    # Base score of the candidate is its popularity
    candidates['score'] = candidates['track_popularity']

    candidates.loc[candidates['primary_artist_name'].isin(liked_artists), 'score'] += 30
    candidates.loc[candidates['ab_genre_rosamerica_value'].isin(liked_genres), 'score'] += 15

    recs = candidates.sort_values('score', ascending=False).head(topk)
    return list(zip(recs['track_id'], recs['track_name']))

# returns a list of tuples of (track_ids, track_name)
def recommend_global_hits(seen_ids, topk):
    candidates = TRACKS[~TRACKS['track_id'].isin(seen_ids)]
    recs = candidates.sort_values('track_popularity', ascending=False).head(topk)
    return list(zip(recs['track_id'], recs['track_name']))


def recommend_probabilistic(user_history, topk=5):
    seen_ids = {x[0] for x in user_history}

    # Filter candidates
    candidates = TRACKS[~TRACKS['track_id'].isin(seen_ids)].copy()

    # Integration of Part 1

    # Map Global Probabilities
    # If an artist is missing , fallback to global P(5-star)
    candidates['p_artist'] = candidates['primary_artist_name'].map(ARTIST_PROBS).fillna(GLOBAL_P5)
    candidates['p_genre'] = candidates['ab_genre_rosamerica_value'].map(GENRE_PROBS).fillna(GLOBAL_P5)

    # Base Utility = Combination of probabilities
    candidates['utility'] = (candidates['p_artist'] + candidates['p_genre']) / 2

    # Get the liked ids
    liked_ids = {x[0] for x in user_history if x[2] >= 4}

    if liked_ids:
        liked_meta = TRACKS[TRACKS['track_id'].isin(liked_ids)]
        liked_artists = set(liked_meta['primary_artist_name'].unique())

        # for artists the user has validated.
        candidates.loc[candidates['primary_artist_name'].isin(liked_artists), 'utility'] *= 2.0

        # Integration of Part 2

        user_avg_dur = liked_meta['duration_ms'].mean()

        if user_avg_dur < MEDIAN_DURATION_MS:
            # Pickier users (prefer short songs) -> Penalize long songs harder
            candidates.loc[candidates['duration_ms'] > MEDIAN_DURATION_MS, 'utility'] *= 0.8
        else:
            # Patient users -> Slight boost to longer songs
            candidates.loc[candidates['duration_ms'] >= MEDIAN_DURATION_MS, 'utility'] *= 1.1



        # Only boost party songs if the user likes party songs.
        if 'ab_mood_party_value' in liked_meta.columns:
            # Check if user has liked any song labeled as 'party'
            user_likes_party = 'party' in liked_meta['ab_mood_party_value'].values

            # Apply boost only if they are a "party song lover"
            if user_likes_party:
                if 'ab_mood_party_value' in candidates.columns:
                    candidates.loc[candidates['ab_mood_party_value'] == 'party', 'utility'] *= 1.1


    # Adding an exponent makes the peaks sharper
    candidates['sampling_weight'] = candidates['utility'] ** 3
    candidates['prob'] = candidates['sampling_weight'] / candidates['sampling_weight'].sum()

    # Weighted random choice
    recs = candidates.sample(n=topk, weights='prob')
    return list(zip(recs['track_id'], recs['track_name']))


# song_ratings: List of (track_id, track_name, rating), topk: Number of songs to return
# Decides on the recommendation format to use
def query(song_ratings, topk):

    num_ratings = len(song_ratings)

    # Check if they have found a 5-star yet
    has_five_star = any(r == 5 for _, _, r in song_ratings)

    # New user (< 5 ratings) or unhappy user (No 5 stars yet)
    if num_ratings < 5 or not has_five_star:
        return recommend_safe(song_ratings, topk)

    # Happy/experienced user
    else:
        return recommend_probabilistic(song_ratings, topk)

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('data/melih_ratings.csv')
    # Sort by 'round_idx'
    df_sorted = df.sort_values('round_idx')
    #Create the list of tuples: (song_id, track_name, rating)
    history = list(df_sorted[['song_id', 'track_name', 'rating']].itertuples(index=False, name=None))

    print("\n--- Model 1: Safe Recommender ---")
    for r in recommend_safe(history, 5):
        print(r)

    print("\n--- Model 2: Probabilistic Recommender ---")
    for r in recommend_probabilistic(history, 5):
        print(r)

    print("\n--- Automatic Query Decision ---")
    for r in query(history, 5):
        print(r)
