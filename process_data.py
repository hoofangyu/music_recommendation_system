import pandas as pd
from src.process_data.process_data_helpers import clean_data, scale_numerical_features, cluster_popularity, cluster_duration_ms, embed_track_genre
from src.process_data.process_data_lyrics import generate_song_lyrics, embed_lyrics

def process_data_from_csv(filepath):
    df = pd.read_csv(filepath)
    df = clean_data(df)
    df = scale_numerical_features(df)
    df = cluster_popularity(df)
    df = cluster_duration_ms(df)
    df = embed_track_genre(df)
    print("Phase 1 of data processing done, moving on to phase 2: Obtaining Lyrics")
    df = generate_song_lyrics(df)
    df.to_csv("data/processed/dataset_w_lyrics.csv")
    df = embed_lyrics(df)
    print("Dataset has been parsed successfully!")
    print("Below is a peak of the processed dataset!")
    print(df.head())
    return df

if __name__ == "__main__":
    df = process_data_from_csv("data/raw/dataset.csv")
    df.to_csv("data/processed/dataset_parsed.csv")