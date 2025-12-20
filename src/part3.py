# Part 3: Recommender Design

"""
Part 3: Recommender Design

Design and implement two different recommendation algorithms for the music system.
"""

import pandas as pd
import numpy as np
import random

print("Initializing Recommender.")
try:
    # Constant from Part 2 Findings
    MEDIAN_DURATION_MS = 233712
    #read all tracks
    TRACKS = pd.read_csv("../data/tracks.csv")
    # Map track_id -> popularity
    TRACK_SCORES = TRACKS.set_index('track_id')['track_popularity'].to_dict()

    # 2. Artist & Genre Average Scores
    ARTIST_SCORES = TRACKS.groupby('primary_artist_name')['track_popularity'].mean().to_dict()
    GENRE_SCORES = TRACKS.groupby('ab_genre_rosamerica_value')['track_popularity'].mean().to_dict()
except FileNotFoundError:
    print("WARNING: Could not load data/tracks.csv. Recommender will fail.")
    TRACKS = pd.DataFrame()

# user history: List[Tuple[str, str, int]]
#                   (track_id, track_name, rating)
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
    seen_ids = set()
    for x in user_history:
        seen_ids.add(x[0])
    candidates = TRACKS[~TRACKS['track_id'].isin(seen_ids)].copy()

    candidates['utility'] = candidates['track_popularity'].astype(float).pow(0.7)

    liked_ids = set()
    for x in user_history:
        if x[2] >= 4:
            liked_ids.add(x[0])

    if liked_ids:
        liked_meta = TRACKS[TRACKS['track_id'].isin(liked_ids)]

        if not liked_meta.empty:
            user_avg_dur = liked_meta['duration_ms'].mean()

            # Artist boost (Strongest Signal)
            liked_artists = set(liked_meta['primary_artist_name'].dropna().unique())
            # If user likes the artist, triple the score
            candidates.loc[candidates['primary_artist_name'].isin(liked_artists), 'utility'] *= 3.0

            if user_avg_dur < MEDIAN_DURATION_MS:
                # people who prefer short songs are pickier, therefore they have a higher muliplier
                candidates.loc[candidates['duration_ms'] < MEDIAN_DURATION_MS, 'utility'] *= 1.5
            else:
                # people who prefer long songs are less picky, nevertheless their pick is acknowledged
                candidates.loc[candidates['duration_ms'] >= MEDIAN_DURATION_MS, 'utility'] *= 1.2


    # Boost party songs since party song lovers are found to be picky in the second part
    candidates.loc[candidates['ab_mood_party_value'] == 'party', 'utility'] *= 1.1


    # Add small epsilon to avoid 0 division or errors
    candidates['prob'] = (candidates['utility'] + 1) / (candidates['utility'] + 1).sum()



    # ---------------------------------------------------------
    # DEBUG SECTION: PRINT & SAVE PROBABILITIES
    # ---------------------------------------------------------
    #print("\n--- Top 10 Songs by Calculated Probability ---")
    #debug_view = candidates[['track_name', 'track_popularity', 'utility', 'prob']]
    #print(debug_view.sort_values('prob', ascending=False).head(10).to_string(index=False))

    # Optional: Save all probabilities to a CSV to inspect in Excel
    #debug_view.sort_values('prob', ascending=False).to_csv("probability_debug.csv", index=False)
    #print("\n(Full probability list saved to 'probability_debug.csv')")
    # ---------------------------------------------------------



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
    # 1. Read the CSV file
    df = pd.read_csv('../data/melih_ratings.csv')
    # 2. Sort by 'round_idx' to ensure the history is in the correct chronological order
    df_sorted = df.sort_values('round_idx')
    # 3. Create the list of tuples: (song_id, track_name, rating)
    history = list(df_sorted[['song_id', 'track_name', 'rating']].itertuples(index=False, name=None))


    print("\nTesting Query Function...")
    results = query(history, 5)
    print("Recommendations:")
    for r in results:
        print(r)
