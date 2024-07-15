import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from access_tokens import GENIUS_ACCESS_TOKEN
from langchain_aws import BedrockEmbeddings
import lyricsgenius as lg
import pandas as pd
import string
import re
import requests

def clean_string(input_string):
    translation_table = str.maketrans('', '', string.punctuation)
    no_punctuation_string = input_string.translate(translation_table)
    cleaned_string = re.sub(r'\s+', ' ', no_punctuation_string).strip()
    
    return cleaned_string

def generate_lyrics_url(artist_col, track_name_col):
    # Create URL to Genius
    url = []
    for i in range(len(artist_col)):
        artist_str = artist_col[i].lower().replace(";"," and ")
        track_name_str = track_name_col[i].lower()
        artist_str = clean_string(artist_str)
        track_name_str = clean_string(track_name_str)
        final_str = artist_str + '-' + track_name_str
        final_str = final_str.replace(" ","-").capitalize() + '-lyrics'
        url.append(final_str)
    
    return url

def adjust_lyrics(input_string):
    # Clean Scraped Lyrics
    position = input_string.find("Lyrics")
    if position != -1:
        result_string = input_string[position + len("Lyrics"):]
        result_string = result_string.strip()
    else:
        result_string = input_string
    
    result_string = re.sub(r'\d+Embed$', '', result_string).strip()

    return result_string

def generate_song_lyrics(df):
    # Scrap Genius for Lyrics and add to Dataframe
    genius_access_token = GENIUS_ACCESS_TOKEN
    genius = lg.Genius(genius_access_token)

    print("Beginning Lyric Scraping process...")

    artist_col = df['artists']
    track_name_col = df['track_name']
    url_list = generate_lyrics_url(artist_col,track_name_col)
    lyrics_list = []
    length_df = len(df)
    count = 1

    for url in url_list:
        try:
            lyrics = genius.lyrics(song_url=url)
            lyrics = adjust_lyrics(lyrics)
        except requests.exceptions.HTTPError as http_err:
            lyrics = "Lyrics not found on Genius"
        except Timeout:
            lyrics = "Timeout"
        finally:
            lyrics_list.append(lyrics)
    
        print(f"Lyrics obtained: {count}/{len(df)}")
        count += 1
    
    df['lyrics'] = lyrics_list
    print("Lyrics all obtained, proceeding to embed lyrics!")
    return df

def embed_lyrics(df):
    # embed song lyrics into vector
    # if song lyrics not found, use song name instead
    embeddings_list = []
    embeddings = BedrockEmbeddings()

    length_df = len(df)
    print("Beginning Lyric Embedding process...")

    for i in range(length_df):
        lyrics = df['lyrics'][i]
        if  lyrics == "Lyrics not found on Genius" or lyrics == "Timeout":
            embedding = embeddings.embed_query(df['track_name'][i])
        else:
            embedding = embeddings.embed_query(lyrics)
        embeddings_list.append(embedding)
        print(f"Lyrics embedded: {i+1}/{len(df)}")
    
    print("Lyrics all obtained, proceeding to embed lyrics!")

    embeddings_df = pd.DataFrame(embeddings_list, columns=[f'lyrics_{i}' for i in range(len(embeddings_list[0]))])
    df_with_embeddings = pd.concat([df, embeddings_df], axis=1)
    return df_with_embeddings