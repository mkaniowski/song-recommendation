import ast

import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class SongRecommender:
    def __init__(self, csv_path, n_artist_features=10):
        # Load the main song dataset.
        self.data = pd.read_csv(csv_path)

        # Define the numeric features used for similarity.
        self.numeric_features = [
            'danceability', 'energy', 'loudness', 'tempo',
            'duration_ms', 'acousticness', 'speechiness', 'year'
        ]

        # Normalize the numeric features.
        scaler = MinMaxScaler()
        self.data[self.numeric_features] = scaler.fit_transform(self.data[self.numeric_features])

        # Extract a representative artist value from the stored artist list.
        self.data['main_artist'] = self.data['artists'].apply(self._extract_main_artist)

        # Use FeatureHasher to encode the main artist into a fixed-size numeric vector.
        # Wrap each main artist string in a list so that the input is an iterable of iterables.
        self.n_artist_features = n_artist_features
        hasher = FeatureHasher(n_features=self.n_artist_features, input_type='string')
        artist_hashed = hasher.transform(self.data['main_artist'].apply(lambda x: [x])).toarray()
        # Create column names for these new features.
        artist_feature_cols = [f'artist_feature_{i}' for i in range(self.n_artist_features)]
        # Put the hashed features into a DataFrame.
        artist_features_df = pd.DataFrame(artist_hashed, columns=artist_feature_cols, index=self.data.index)
        # Concatenate these new artist features with the normalized numeric features.
        self.data = pd.concat([self.data, artist_features_df], axis=1)

        # Update the list of features used for similarity to include the hashed artist features.
        self.features = self.numeric_features + artist_feature_cols

        # Placeholder for the similarity data (e.g., playlists).
        self.sim_data = None

    @staticmethod
    def _extract_main_artist(artist_str):
        """
        Given a string like "['Eminem']" or "['Adele', 'Someone']",
        extract and return the first artist. In case of an error, return an empty string.
        """
        try:
            artist_list = ast.literal_eval(artist_str)
            if isinstance(artist_list, list) and len(artist_list) > 0:
                return artist_list[0]
        except Exception:
            pass
        return ""

    def recommend(self, song_names, top_n=1):
        """
        For each song in song_names, compute cosine similarity using the combined feature vector
        (numeric features + hashed artist features) with the entire dataset (excluding the input song)
        and return the top_n recommendations.
        """
        global_recommendations = []
        for song_name in song_names:
            print(f"Processing recommendations for: {song_name}")
            # Find the song in the dataset.
            song_data = self.data[self.data['name'] == song_name]
            if song_data.empty:
                continue

            # Use only the first match.
            song_data = song_data.iloc[[0]]
            input_vector = song_data[self.features].values.reshape(1, -1)

            # Compute cosine similarity using the combined feature vector.
            similarity = cosine_similarity(input_vector, self.data[self.features])
            temp_data = self.data.copy()
            temp_data['similarity'] = similarity[0]

            # Exclude the input song.
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
        """
        Load the playlist/similarity data. The dataframe is expected to have at least:
          - 'playlistname'
          - 'trackname'
        """
        self.sim_data = pd.read_csv(csv_path)

    def simulate(self, top_n=5, removal_count=5):
        """
        Simulator workflow:
          1. Select a random playlist from sim_data.
          2. Randomly remove `removal_count` songs from that playlist.
          3. Use the remaining songs (limited to 30) as input; for each, get `top_n`
             recommendations from the full dataset (using the combined feature vector).
          4. Combine all recommended songs (unique by song name).
          5. For each recommended song, compute its cosine similarity against each removed song
             using the combined feature vector. Record the maximum similarity and which removed song
             gave that similarity.
          6. Return the top recommended songs (with highest similarity) along with details.
        """
        if self.sim_data is None:
            raise ValueError("Similarity data not loaded. Please load the similarity data first.")

        # Step 1: Select a random playlist.
        random_playlist = self.sim_data['playlistname'].sample(n=1).iloc[0]
        playlist_songs_df = self.sim_data[self.sim_data['playlistname'] == random_playlist]

        # Step 2: Randomly remove removal_count songs from the playlist.
        removed_songs = playlist_songs_df['trackname'].sample(n=removal_count).tolist()

        # Step 3: Use the remaining songs as input (limit to 30 songs).
        reduced_playlist_df = playlist_songs_df[~playlist_songs_df['trackname'].isin(removed_songs)]
        if len(reduced_playlist_df) > 30:
            reduced_playlist_df = reduced_playlist_df.sample(30)
        input_song_names = reduced_playlist_df['trackname'].tolist()

        # Step 4: Get recommendations for each input song.
        global_recs = self.recommend(input_song_names, top_n=top_n)

        # Step 5: Combine all recommended songs uniquely.
        recommended_songs = {}
        for rec in global_recs:
            for song in rec['recommendations']:
                song_name = song['recommended_song']
                if song_name not in recommended_songs:
                    recommended_songs[song_name] = song
        recommended_song_names = list(recommended_songs.keys())

        # Get the combined feature vectors for the removed songs.
        removed_subset = self.data[self.data['name'].isin(removed_songs)]
        if removed_subset.empty:
            raise ValueError("None of the removed songs were found in the main dataset.")

        # Get the combined feature vectors for the recommended songs.
        rec_subset = self.data[self.data['name'].isin(recommended_song_names)]
        if rec_subset.empty:
            raise ValueError("No recommended songs found in the main dataset.")

        rec_features = rec_subset[self.features].values  # shape: (num_recs, combined_dim)
        removed_features = removed_subset[self.features].values  # shape: (num_removed, combined_dim)

        # Compute cosine similarity between each recommended song and each removed song.
        similarity_matrix = cosine_similarity(rec_features, removed_features)
        # For each recommended song, take the maximum similarity value.
        aggregated_sim = similarity_matrix.max(axis=1)
        max_indices = similarity_matrix.argmax(axis=1)
        removed_song_names_arr = removed_subset['name'].values
        compared_to = [removed_song_names_arr[idx] for idx in max_indices]

        # Attach the aggregated similarity and the removed song used for comparison.
        rec_subset = rec_subset.copy()
        rec_subset['aggregated_similarity'] = aggregated_sim
        rec_subset['compared_to'] = compared_to

        # Sort by aggregated similarity (descending) and take the top_n.
        top_rec_df = rec_subset.sort_values(by='aggregated_similarity', ascending=False).head(top_n)
        top_recommendations = []
        for _, row in top_rec_df.iterrows():
            top_recommendations.append({
                'recommended_song': row['name'],
                'artists': row['artists'],
                'aggregated_similarity': row['aggregated_similarity'],
                'compared_to': row['compared_to']
            })

        # Return the simulation details.
        return {
            'playlist': random_playlist,
            'removed_songs': removed_songs,
            'input_songs': input_song_names,
            'global_recommendations': global_recs,
            'top_recommendations': top_recommendations
        }

# Example usage:
# recommender = SongRecommender('songs.csv')
# recommender.load_sim_data('sim_data.csv')
# results = recommender.simulate(top_n=3, removal_count=5)
# print(results)
