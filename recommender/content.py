from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Spotify parçaları veri setini yükle
dataset = load_dataset("maharshipandya/spotify-tracks-dataset") 
dataframe = dataset['train']
df = dataframe.to_pandas()

# 'Unnamed: 0' sütununu kaldır
df = df.drop('Unnamed: 0', axis=1)

print('The music dataset has', df["artists"].nunique(), 'unique artists')
print('The music dataset has', df["popularity"].nunique(), 'unique popularity')
print('The music dataset has', df["energy"].nunique(), 'unique energy')

print('The unique popularity values are', sorted(df['popularity'].unique()))

# İlgili sütunları 'tags' içinde birleştir
tfidf = TfidfVectorizer(stop_words="english")
tags = df.artists + ' ' + df.album_name + ' ' + df.track_genre
df['tags'] = tags
new_df = df[['track_name', 'tags', 'popularity']]

# Print the head of the new DataFrame
print(new_df.head())

# Fill missing values in 'tags' column
new_df['tags'] = new_df['tags'].fillna('')

# Sample a subset of the DataFrame
sample_size = min(100, len(new_df))  # Ensure sample size is valid
sampled_df = new_df.sample(n=sample_size, random_state=42)

# Print the sampled DataFrame
print(sampled_df.head(10))

# Generate TF-IDF matrix for the sampled data
tfidf_matrix_sampled = tfidf.fit_transform(sampled_df['tags'])

# Print the shape of the TF-IDF matrix for the sampled data
print("TF-IDF matrix shape for sampled data:", tfidf_matrix_sampled.shape)

# Calculate cosine similarity using the sampled TF-IDF matrix
cosine_sim_sampled = cosine_similarity(tfidf_matrix_sampled, tfidf_matrix_sampled)

# Print the shape of the cosine_sim_sampled array
print("Shape of cosine_sim_sampled array:", cosine_sim_sampled.shape)

# Create indices Series for sampled DataFrame
indices = pd.Series(sampled_df.index, index=sampled_df['track_name'])
indices = indices[~indices.index.duplicated(keep='last')]


example_track_name = "Failed Organum"

# Check if the track name is in indices
if example_track_name in indices.index:
    # Get the index of the example track for sampled DataFrame
    track_index = indices[example_track_name]
    # Make sure the index is within the array dimensions
    if track_index >= len(cosine_sim_sampled):
        track_index = len(cosine_sim_sampled) - 1 
        
    # Calculate similarity scores using the sampled cosine_sim matrix
    similarity_scores = pd.DataFrame(cosine_sim_sampled[track_index], columns=["similarity"])
    
    # Combine similarity scores with popularity values
    result_df = pd.concat([similarity_scores,sampled_df[['popularity']].reset_index(drop=True)],axis=1)
    
    # Get indices of top similar tracks
    recommended_indices = similarity_scores.sort_values("similarity", ascending=False)[1:11].index
    
    # Print recommended tracks along with similarity and popularity
    recommended_tracks = sampled_df.iloc[recommended_indices]
    recommended_tracks = pd.concat([recommended_tracks, similarity_scores.iloc[recommended_indices]], axis=1)
    
    print(result_df.head())
    print(recommended_tracks[['track_name', 'similarity', 'popularity']])
else:
    print("Example track not found in indices.")



