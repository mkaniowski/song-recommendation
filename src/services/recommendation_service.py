import joblib
import pandas as pd

class Recommendation_service:
    def __init__(self):
        self.cosine = joblib.load('E:/Michal/Dokumenty/Projekty/song-recommendation/notebooks/cosine_sim_model.pkl')
        # self.tfidf = joblib.load('E:/Michal/Dokumenty/Projekty/song-recommendation/notebooks/tfidf_model.pkl')
        self.songs_df = pd.read_csv('E:\Michal\Dokumenty\Projekty\song-recommendation\data\sampled_songs.csv')

    def get_recommendations(self, playlist, last_top=10):
        # Validate input: Check if all songs in the playlist exist in the dataset
        missing_songs = [song for song in playlist if song not in self.songs_df['name'].values]
        if missing_songs:
            return f"The following songs are not in the dataset: {', '.join(missing_songs)}"

        # Get indices of all songs in the playlist
        playlist_indices = [
            self.songs_df[self.songs_df['name'] == song].index[0]
            for song in playlist
        ]

        # Initialize similarity scores
        aggregated_scores = [0] * len(self.songs_df)

        # Accumulate similarity scores for all songs in the playlist
        for idx in playlist_indices:
            aggregated_scores = [agg + sim for agg, sim in zip(aggregated_scores, self.cosine[idx])]

        # Normalize by the size of the playlist
        aggregated_scores = [score / len(playlist) for score in aggregated_scores]

        # Create a list of song indices with their similarity scores
        sim_scores = list(enumerate(aggregated_scores))

        # Sort the songs based on the aggregated similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Filter out songs already in the playlist
        sim_scores = [score for score in sim_scores if score[0] not in playlist_indices]

        # Get the top N most similar songs
        sim_scores = sim_scores[:last_top]

        # Get the song indices
        song_indices = [i[0] for i in sim_scores]

        # Return the top N most similar songs
        return self.songs_df.iloc[song_indices]