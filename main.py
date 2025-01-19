import os
import shutil
from http.client import HTTPException
from typing import List

import kagglehub
from fastapi import FastAPI
from pydantic import BaseModel

from src.services.SongRecommender import SongRecommender
from src.services.recommendation_service import Recommendation_service

# Define the path to the data folder
data_folder = os.path.abspath('../data/')

# Check if the data folder exists and is not empty
if not os.path.exists(data_folder) or not os.listdir(data_folder):
    # Create the data folder if it does not exist
    os.makedirs(data_folder, exist_ok=True)

    # Download the latest version of the dataset without specifying the path
    path = kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs", force_download=True)
    print("Path to downloaded dataset files:", path)

    # Move the downloaded files to the data folder
    for file_name in os.listdir(path):
        full_file_name = os.path.join(path, file_name)
        if os.path.isfile(full_file_name):
            shutil.move(full_file_name, data_folder)
else:
    print("Data folder already contains files.")


app = FastAPI()



recommender = SongRecommender('E:/Michal/Dokumenty/Projekty/song-recommendation/data/tracks_features.csv')
rand_songs = recommender.data['name'].sample(3)
# recommendations = recommender.recommend(rand_songs.values, 5)
# print(recommendations)


class RecommendationRequest(BaseModel):
    song_names: List[str]
    num_recommendations: int = 5

@app.post("/recommend")
async def recommend_songs(request: RecommendationRequest):
    try:
        recommendations = recommender.recommend(request.song_names, request.num_recommendations)
        return {"recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
