from src.services.SongRecommender import SongRecommender
import json

recommender = SongRecommender('E:/Michal/Dokumenty/Projekty/song-recommendation/data/tracks_features.csv')
recommender.load_sim_data('E:/Michal/Dokumenty/Projekty/song-recommendation/data_sim/playlists_filtered.csv')


def main():
    # Get the recommendations
    recoms = recommender.simulate(top_n=3)

    # Extract components from the result
    global_recommendations = recoms['global_recommendations']
    removed_recs = recoms['removed_songs_recommendations']
    playlist = recoms['playlist']
    removed_songs = recoms['removed_songs']

    # Print header information
    print("=" * 50)
    print(f"Playlist: {playlist}")
    print(f"Removed Songs: {', '.join(removed_songs)}")
    print("=" * 50)
    print("\nGLOBAL RECOMMENDATIONS:\n")

    # Print Global Recommendations
    for rec in global_recommendations:
        print("-" * 50)
        print(f"Input Song: {rec['input_song']}")
        for idx, r in enumerate(rec['recommendations'], start=1):
            print(f"  {idx}. {r['recommended_song']} by {r['artists']} (Similarity: {r['similarity']})")
        print("-" * 50)
        print()

    print("\nREMOVED SONGS RECOMMENDATIONS:\n")
    # Print Recommendations from Removed Songs
    for rec in removed_recs:
        print("-" * 50)
        print(f"Input Song: {rec['input_song']}")
        for idx, r in enumerate(rec['recommendations'], start=1):
            print(f"  {idx}. {r['recommended_song']} by {r['artists']} (Similarity: {r['similarity']})")
        print("-" * 50)
        print()


if __name__ == '__main__':
    main()
