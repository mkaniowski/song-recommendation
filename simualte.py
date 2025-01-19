import os
import shutil

import kagglehub
import pandas as pd
import requests

data_folder = os.path.abspath('./data_sim/')

# Check if the data folder exists and is not empty
if not os.path.exists(data_folder) or not os.listdir(data_folder):
    # Create the data folder if it does not exist
    os.makedirs(data_folder, exist_ok=True)

    # Download the latest version of the dataset without specifying the path
    path = kagglehub.dataset_download("andrewmvd/spotify-playlists", force_download=True)
    print("Path to downloaded dataset files:", path)

    # Move the downloaded files to the data folder
    for file_name in os.listdir(path):
        full_file_name = os.path.join(path, file_name)
        if os.path.isfile(full_file_name):
            shutil.move(full_file_name, data_folder)
else:
    print("Data folder already contains files.")

# Load the data
songs_df = pd.read_csv('E:/Michal/Dokumenty/Projekty/song-recommendation/data/tracks_features.csv')

# if there is no file ./data_sim/playlists_filtered.csv' then print ok
if not os.path.exists('./data_sim/playlists_filtered.csv'):
    # Load the data
    print("Loading data...")
    playlists_df = pd.read_csv('E:/Michal/Dokumenty/Projekty/song-recommendation/data_sim/spotify_dataset.csv', on_bad_lines='skip')

    # Clean column names by stripping extra spaces and quotation marks
    playlists_df.columns = playlists_df.columns.str.replace('"', '').str.strip()
    songs_df.columns = songs_df.columns.str.replace('"', '').str.strip()

    # Normalize column names
    playlists_df.columns = playlists_df.columns.str.lower()
    songs_df.columns = songs_df.columns.str.lower()

    # Ensure data types are compatible for filtering
    playlists_df['trackname'] = playlists_df['trackname'].astype(str)
    songs_df['name'] = songs_df['name'].astype(str)

    # Filter rows where trackname is in the songs_df 'name' column
    filtered_playlists_df = playlists_df[playlists_df['trackname'].isin(songs_df['name'])]

    # Save the filtered data
    filtered_playlists_df.to_csv('./data_sim/playlists_filtered.csv', index=False)
    print("Filtered playlists saved successfully.")

else:
    print("Filtered playlists already exist.")
    # filtered_playlists_df = pd.read_csv('./data_sim/playlists_filtered.csv')


def main(num_recommendations=5):
    # Load the left joined playlists
    filtered_playlists_df = pd.read_csv('./data_sim/playlists_filtered.csv')

    # Select a random playlist
    random_playlist = filtered_playlists_df['playlistname'].sample(n=1).iloc[0]

    # Select all songs from the random playlist
    playlist_songs_df = filtered_playlists_df[filtered_playlists_df['playlistname'] == random_playlist]

    # Select a random subset of songs from the playlist
    random_songs = playlist_songs_df['trackname'].sample(n=num_recommendations).tolist()

    # Exclude the random songs from the playlist
    reduced_playlists_df = playlist_songs_df[~playlist_songs_df['trackname'].isin(random_songs)]

    # Get the song names
    song_names = reduced_playlists_df['trackname'].values.tolist()

    # Make a POST request to the recommendation service
    data = {
        'song_names': song_names,
        'num_recommendations': num_recommendations
    }

    # Make a POST request to the recommendation service
    r = requests.post('http://localhost:8000/recommend', json=data)
    print("Removed songs:", random_songs)
    print("Recommended songs:", r.json()['recommendations'])


# Example usage
main(5)