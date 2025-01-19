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
            'duration_ms', 'acousticness', 'speechiness'
        ]

        # Normalize the features
        scaler = MinMaxScaler()
        self.data[self.features] = scaler.fit_transform(self.data[self.features])

    def recommend(self, song_names, num_recommendations=5):
        # Filter songs that match the given names
        song_data = self.data[self.data['name'].isin(song_names)]

        if song_data.empty:
            raise ValueError("None of the provided song names are found in the dataset.")

        # Compute the mean feature vector of the input songs
        input_vector = song_data[self.features].mean(axis=0).values.reshape(1, -1)

        # Compute cosine similarity between input vector and all songs
        similarity = cosine_similarity(input_vector, self.data[self.features])

        # Add similarity scores to the dataset
        self.data['similarity'] = similarity[0]

        # Exclude input songs from recommendations
        recommendations = self.data[~self.data['name'].isin(song_names)]

        # Sort by similarity and select top recommendations
        top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(num_recommendations)

        return top_recommendations[['name', 'artists', 'similarity']].to_dict(orient='records')