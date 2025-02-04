import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class SongRecommender:
    def __init__(self, csv_path):
        # Load the dataset
        self.data = pd.read_csv(csv_path)

        # Select the feature columns
        self.features = [
            'danceability', 'energy', 'loudness', 'tempo',
            'duration_ms', 'acousticness', 'speechiness', 'year'
        ]

        # Normalize the features
        scaler = MinMaxScaler()
        self.data[self.features] = scaler.fit_transform(self.data[self.features])
        self.sim_data = None

    def recommend(self, song_names, top_n=1):
        print("Recommendation started...")
        """
        For each song in song_names, compute cosine similarity with the entire dataset
        (excluding the input song) and return the top_n recommendations.
        """
        global_recommendations = []
        for song_name in song_names:
            print(f"Searching recommendations for: {song_name}")
            # Find the song in the dataset
            song_data = self.data[self.data['name'] == song_name]

            if song_data.empty:
                # If the song is not found, skip it.
                continue

            # Use only the first match to get the feature vector.
            song_data = song_data.iloc[[0]]  # DataFrame format

            # Get feature vector for the current song
            input_vector = song_data[self.features].values.reshape(1, -1)

            # Compute cosine similarity between the input song and all songs
            similarity = cosine_similarity(input_vector, self.data[self.features])

            # Create a temporary copy to add similarity scores.
            temp_data = self.data.copy()
            temp_data['similarity'] = similarity[0]

            # Exclude the input song from recommendations.
            filtered_data = temp_data[temp_data['name'] != song_name]

            # Get the top_n most similar songs.
            top_recs = filtered_data.sort_values(by='similarity', ascending=False).head(top_n)

            rec_list = []
            for _, row in top_recs.iterrows():
                rec_list.append({
                    'recommended_song': row['name'],
                    'artists': row['artists'],
                    'similarity': row['similarity']
                })

            global_recommendations.append({
                'input_song': song_name,
                'recommendations': rec_list
            })

        return global_recommendations

    def load_sim_data(self, csv_path):
        print("Loading simulator data...")
        self.sim_data = pd.read_csv(csv_path)

    def simulate(self, top_n=5):
        print("Simulation started...")
        """
        This method performs the following steps:
          1. Select a random playlist from sim_data.
          2. Randomly remove 'top_n' songs from the playlist (these become the "removed songs").
          3. For every other song (input songs) in the playlist, compute:
              a. Global recommendations from the full dataset (using recommend()).
              b. Recommendations among the removed songs only (by computing cosine similarity against only those songs).
          4. Return a dictionary containing two lists of recommendation results.
        """
        if self.sim_data is None:
            raise ValueError("Similarity data not loaded. Please load the similarity data first.")

        # Step 1: Select a random playlist.
        random_playlist = self.sim_data['playlistname'].sample(n=1).iloc[0]

        # Step 2: Get all songs from the random playlist.
        playlist_songs_df = self.sim_data[self.sim_data['playlistname'] == random_playlist]

        if len(playlist_songs_df) > 20:
            playlist_songs_df = playlist_songs_df.sample(n=20)

        # Randomly select 'top_n' songs to remove.
        removed_songs = playlist_songs_df['trackname'].sample(n=top_n).tolist()

        # Step 3: The remaining songs will serve as input songs.
        reduced_playlist_df = playlist_songs_df[~playlist_songs_df['trackname'].isin(removed_songs)]
        input_song_names = reduced_playlist_df['trackname'].values.tolist()

        # (A) Global recommendations from the full dataset.
        global_recs = self.recommend(input_song_names, top_n=1)

        # (B) Recommendations computed only among the removed songs.
        removed_recs = []
        for song_name in input_song_names:
            # Find the input song in the main dataset.
            song_data = self.data[self.data['name'] == song_name]
            if song_data.empty:
                continue
            song_data = song_data.iloc[[0]]
            input_vector = song_data[self.features].values.reshape(1, -1)

            # Limit self.data to only the removed songs.
            removed_subset = self.data[self.data['name'].isin(removed_songs)]
            if removed_subset.empty:
                # If none of the removed songs are in the main dataset, skip.
                continue

            # Compute cosine similarity between the input song and the removed songs.
            similarity = cosine_similarity(input_vector, removed_subset[self.features])
            temp_removed = removed_subset.copy()
            temp_removed['similarity'] = similarity[0]

            # Get the top_n recommendations among the removed songs.
            top_removed = temp_removed.sort_values(by='similarity', ascending=False).head(top_n)

            rec_list = []
            for _, row in top_removed.iterrows():
                rec_list.append({
                    'recommended_song': row['name'],
                    'artists': row['artists'],
                    'similarity': row['similarity']
                })

            removed_recs.append({
                'input_song': song_name,
                'recommendations': rec_list
            })

        # Return both recommendation sets.
        return {
            'global_recommendations': global_recs,
            'removed_songs_recommendations': removed_recs,
            'removed_songs': removed_songs,  # Optionally return the list of removed songs.
            'playlist': random_playlist  # And the playlist name.
        }

# Example usage:
# recommender = SongRecommender('songs.csv')
# recommender.load_sim_data('sim_data.csv')
# results = recommender.simulate(top_n=3)
# print(results)
