# Assignment 16: Anime Recommendation System Using Cosine Similarity

## PART 1: Jupyter Notebook Content

# Title Section

```python
# Assignment 16: Anime Recommendation System Using Cosine Similarity
# Student Name: Ashar Khan
# Batch: Data Science / AI & ML
# Tool Used: Jupyter Notebook
```

---

## Step 1: Import Required Libraries

### Markdown Explanation

In this step, we import all the important libraries required for the project. Pandas and NumPy are used for handling and processing data. Matplotlib and Seaborn are used for creating graphs and visualizations. Scikit-learn is used for converting text data into numerical form and calculating cosine similarity between anime. Cosine similarity helps us understand how similar two anime are based on their genres, ratings, type, and episodes. These libraries make it easier to build a recommendation system in a clean and organized way.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
```

---

## Step 2: Load the Dataset

### Markdown Explanation

In this step, we load the anime dataset into a pandas DataFrame. A DataFrame is like a table where rows represent anime and columns represent details such as anime title, genre, type, rating, number of episodes, and number of members. After loading the dataset, we check the first few rows to understand how the data looks. This helps us confirm whether the dataset is loaded properly and whether all required columns are available.

```python
df = pd.read_csv('anime.csv')

# Display first 5 rows
df.head()
```

```python
# Check dataset shape
print('Number of Rows:', df.shape[0])
print('Number of Columns:', df.shape[1])
```

---

## Step 3: Understand the Dataset

### Markdown Explanation

Before building the recommendation system, it is important to understand the dataset. In this step, we check column names, data types, missing values, and general information about the data. This helps us identify problems such as empty values, wrong data types, or duplicate records. Understanding the data clearly helps in creating better recommendations because the quality of the final model depends heavily on the quality of the dataset.

```python
# Column names
print(df.columns)
```

```python
# Dataset information
df.info()
```

```python
# Statistical summary
df.describe(include='all')
```

```python
# Missing values
missing_values = df.isnull().sum()
print(missing_values)
```

---

## Step 4: Data Cleaning

### Markdown Explanation

Real-world datasets often contain missing values, duplicates, or incorrect data. In this step, we clean the dataset to improve its quality. Missing values in important columns such as genre, rating, and type are filled with suitable values. Duplicate rows are removed to avoid repeated anime records. We also make sure that numeric columns such as episodes, rating, and members are stored in the correct format. Clean data improves the accuracy of the recommendation system.

```python
# Check duplicate rows
print('Duplicate Rows:', df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)
```

```python
# Fill missing values
df['genre'] = df['genre'].fillna('Unknown')
df['type'] = df['type'].fillna('Unknown')
df['rating'] = df['rating'].fillna(df['rating'].median())
df['members'] = df['members'].fillna(df['members'].median())
```

```python
# Convert episodes column to numeric if needed
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
df['episodes'] = df['episodes'].fillna(df['episodes'].median())
```

```python
# Final missing values check
print(df.isnull().sum())
```

---

## Step 5: Exploratory Data Analysis (EDA)

### Markdown Explanation

Exploratory Data Analysis helps us understand the patterns inside the dataset. We use different graphs to study anime ratings, genres, episode counts, and types. These graphs help us identify which anime categories are most common and which anime have the highest ratings or largest audience. EDA is useful because it gives us a better understanding of the data before building the recommendation system.

### 1. Distribution of Anime Ratings

```python
plt.figure(figsize=(8,5))
sns.histplot(df['rating'], bins=20, kde=True)
plt.title('Distribution of Anime Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```

### Graph Explanation

This graph shows how anime ratings are distributed. Most anime ratings are usually between 6 and 8. Very few anime have extremely low or extremely high ratings. This means that most anime in the dataset are moderately liked by users.

### 2. Top 10 Anime Types

```python
plt.figure(figsize=(8,5))
df['type'].value_counts().head(10).plot(kind='bar')
plt.title('Top Anime Types')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()
```

### Graph Explanation

This graph shows the most common anime types such as TV, Movie, OVA, and Special. TV anime usually appear the most because they are released in larger numbers compared to movies or OVAs.

### 3. Top 10 Genres

```python
from collections import Counter

all_genres = ','.join(df['genre'].dropna()).split(',')
all_genres = [genre.strip() for genre in all_genres]
genre_counts = Counter(all_genres)

top_genres = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count'])
top_genres = top_genres.sort_values(by='Count', ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x='Count', y='Genre', data=top_genres)
plt.title('Top 10 Anime Genres')
plt.show()
```

### Graph Explanation

This graph shows the most popular anime genres in the dataset. Genres such as Action, Comedy, Adventure, Fantasy, and Drama are often the most common. These genres are useful for building recommendations because viewers usually prefer anime with similar genres.

### 4. Correlation Heatmap

```python
numeric_columns = ['episodes', 'rating', 'members']

plt.figure(figsize=(8,5))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

### Graph Explanation

The heatmap shows the relationship between numeric columns such as rating, episodes, and members. A positive value means two columns increase together. For example, anime with more members may also have higher ratings.

---

## Step 6: Feature Engineering

### Markdown Explanation

Feature engineering means creating useful input features for the recommendation system. In this project, we combine genre, type, rating, and episode information into one text feature. This combined feature will help us compare anime more effectively. Since cosine similarity works on numerical data, we convert the combined text into numbers using CountVectorizer. This method counts important words and converts them into a format that the computer can understand.

```python
# Convert numeric columns to string
df['rating'] = df['rating'].astype(str)
df['episodes'] = df['episodes'].astype(str)

# Create combined features
df['combined_features'] = (
    df['genre'] + ' ' +
    df['type'] + ' ' +
    df['rating'] + ' ' +
    df['episodes']
)

# Display sample combined features
df[['name', 'combined_features']].head()
```

---

## Step 7: Convert Text Data into Numerical Format

### Markdown Explanation

Computers cannot directly understand text data like genre names or anime type. To solve this problem, we use CountVectorizer. This method converts text into numbers by counting how many times certain words appear. After converting the text into a numerical format, we can calculate similarity between anime using cosine similarity.

```python
vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['combined_features'])

print(feature_matrix.shape)
```

---

## Step 8: Calculate Cosine Similarity

### Markdown Explanation

Cosine similarity is used to measure how similar two anime are. It compares the features of anime and gives a similarity score between 0 and 1. A score close to 1 means the anime are very similar, while a score close to 0 means they are very different. This is the main step of the recommendation system.

```python
cosine_sim = cosine_similarity(feature_matrix)

print(cosine_sim)
```

---

## Step 9: Build Recommendation Function

### Markdown Explanation

In this step, we create a function that recommends anime based on cosine similarity. The user enters the name of an anime, and the function returns similar anime titles. The recommendations are sorted by similarity score, so the most similar anime appear first.

```python
def recommend_anime(anime_name, similarity_threshold=0.3):
    anime_name = anime_name.lower()
    
    matching_anime = df[df['name'].str.lower() == anime_name]
    
    if matching_anime.empty:
        return 'Anime not found in dataset.'
    
    index = matching_anime.index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_anime = []
    
    for anime_index, score in sorted_scores[1:]:
        if score >= similarity_threshold:
            recommended_anime.append((df.iloc[anime_index]['name'], score))
    
    return recommended_anime[:10]
```

---

## Step 10: Test the Recommendation System

### Markdown Explanation

After creating the recommendation function, we test it using a sample anime title. The system returns a list of similar anime along with their similarity scores. This helps us verify whether the recommendation system is working correctly.

```python
recommendations = recommend_anime('Naruto')

for anime, score in recommendations:
    print(f'Anime: {anime}, Similarity Score: {round(score, 2)}')
```

---

## Step 11: Experiment with Different Similarity Thresholds

### Markdown Explanation

Different similarity thresholds can change the number of recommendations. A low threshold gives more recommendations, while a high threshold gives fewer but more accurate recommendations. Testing different thresholds helps us find the best balance between quantity and quality.

```python
print('Threshold 0.2')
print(recommend_anime('Naruto', similarity_threshold=0.2))

print('\nThreshold 0.5')
print(recommend_anime('Naruto', similarity_threshold=0.5))
```

---

## Step 12: Performance Analysis

### Markdown Explanation

The recommendation system works by comparing anime features and identifying similar anime. It performs well for anime that have strong genre and type information. However, the system may not always understand personal user preferences because it is based only on anime content and not on user history. To improve the system, user ratings and collaborative filtering can also be added in the future.

### Strengths
- Easy to build and understand
- Recommends anime with similar genres and types
- Works well even without user history

### Limitations
- Does not use user behavior or watch history
- Recommendations depend heavily on dataset quality
- Similarity may not always match personal taste

---

## Step 13: Real-World Use Case

### Markdown Explanation

Anime recommendation systems are widely used in streaming platforms and entertainment websites. These systems help users discover new anime based on what they already like. Similar recommendation systems are also used by platforms such as Netflix, YouTube, Spotify, and Amazon to improve user experience and increase engagement.

---

## Step 14: Conclusion

### Markdown Explanation

In this project, we built an anime recommendation system using cosine similarity. We cleaned the dataset, explored the data, created combined features, converted text into numerical form, and calculated similarity scores. The system successfully recommends anime based on similar genres, ratings, types, and episodes. This project is a good example of how machine learning and text processing can be used to build recommendation systems.

---

## Step 15: Interview Questions and Answers

### 1. What is collaborative filtering, and how does it work?

Collaborative filtering is a recommendation technique that suggests items based on the behavior of similar users. For example, if two users like similar anime, then anime liked by one user can be recommended to the other.

### 2. What is the difference between user-based and item-based collaborative filtering?

User-based collaborative filtering recommends items by finding users with similar interests. Item-based collaborative filtering recommends items by finding products or anime that are similar to each other.

### 3. Why is cosine similarity used in recommendation systems?

Cosine similarity is used because it measures the similarity between two items based on their features. It is simple, fast, and works well with text data.

---

