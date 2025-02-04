from src.services.SongRecommender import SongRecommender

def main():
    results = recommender.simulate(top_n=3, removal_count=3)

    print("=" * 60)
    print(f"Playlist: {results['playlist']}")
    print(f"Removed Songs: {', '.join(results['removed_songs'])}")
    print(f"Input Songs: {', '.join(results['input_songs'])}")
    print("=" * 60)
    print("\nGLOBAL RECOMMENDATIONS per Input Song:\n")
    for rec in results['global_recommendations']:
        print(f"Input Song: {rec['input_song']}")
        for idx, r in enumerate(rec['recommendations'], start=1):
            print(f"  {idx}. {r['recommended_song']} by {r['artists']} (Similarity: {r['similarity']})")
        print("-" * 60)

    print("\nTOP RECOMMENDATIONS (Most Similar to Removed Songs):\n")
    for idx, rec in enumerate(results['top_recommendations'], start=1):
        print(f"{idx}. {rec['recommended_song']} by {rec['artists']} "
              f"(Aggregated Similarity: {rec['aggregated_similarity']}, "
              f"Compared to: {rec['compared_to']})")
    print("=" * 60)


if __name__ == '__main__':
    recommender = SongRecommender('E:/Michal/Dokumenty/Projekty/song-recommendation/data/tracks_features.csv')
    recommender.load_sim_data('E:/Michal/Dokumenty/Projekty/song-recommendation/data_sim/playlists_filtered.csv')
    main()

