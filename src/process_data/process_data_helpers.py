from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from langchain_aws import BedrockEmbeddings
import pandas as pd
import joblib

def clean_data(df):
    df = df.iloc[:,1:] 
    df = df.drop_duplicates(['artists','track_name'])
    df = df.dropna().reset_index(drop=True)
    return df

def scale_numerical_features(df):
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_scaled = df.copy()
    
    # Select only numerical columns
    numerical_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns
    
    # Initialize the scaler
    scaler = StandardScaler()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    joblib.dump(scaler, 'data/objects/scaler.pkl')
    
    return df_scaled


def cluster_popularity(df):
    popularity = df[['popularity']]
    optimal_clusters = 4 
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)

    df['popularity_clusters'] = kmeans.fit_predict(popularity)
    cluster_means = df.groupby('popularity_clusters')['popularity'].mean()
    sorted_clusters = cluster_means.sort_values().index
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}
    df['popularity_clusters'] = df['popularity_clusters'].map(cluster_mapping)

    joblib.dump(kmeans, 'data/objects/kmeans_popularity.pkl')
    return df

def cluster_duration_ms(df):
    duration_ms= df[['duration_ms']]
    optimal_clusters = 6 
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
    
    df['duration_clusters'] = kmeans.fit_predict(duration_ms)
    cluster_means = df.groupby('duration_clusters')['duration_ms'].mean()
    sorted_clusters = cluster_means.sort_values().index
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}
    df['duration_clusters'] = df['duration_clusters'].map(cluster_mapping)
    
    joblib.dump(kmeans, 'data/objects/kmeans_duration_ms.pkl')
    return df

def build_embeddings_dictionary(genre_col):
    unique_values = genre_col.unique()
    embeddings_dictionary = {}
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    for genre in unique_values:
        embedding = embeddings.embed_query(genre)
        embeddings_dictionary[genre] = embedding
    
    return embeddings_dictionary

def embed_track_genre(df):
    # build embedding dictionary to save cost and speed up process
    embeddings_dictionary = build_embeddings_dictionary(df['track_genre'])

    embeddings_list = []

    for genre in df['track_genre']:
        embedding = embeddings_dictionary[genre]
        embeddings_list.append(embedding)

    embeddings_df = pd.DataFrame(embeddings_list, columns=[f'genre_{i}' for i in range(len(embeddings_list[0]))])
    df_with_embeddings = pd.concat([df, embeddings_df], axis=1)
    return df_with_embeddings
