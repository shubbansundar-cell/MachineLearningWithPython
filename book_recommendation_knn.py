import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------
# LOAD DATA (already provided in FCC notebook)
# -------------------------------------------------
# In Colab, these are already loaded:
# ratings, books

# For standalone use, uncomment and provide paths:
# ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", encoding="latin-1")
# books = pd.read_csv("BX-Books.csv", sep=";", encoding="latin-1")

# -------------------------------------------------
# DATA CLEANING
# -------------------------------------------------

# Remove users with < 200 ratings
user_counts = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 200].index)]

# Remove books with < 100 ratings
book_counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 100].index)]

# -------------------------------------------------
# CREATE USER-BOOK MATRIX
# -------------------------------------------------
book_user_matrix = ratings.pivot_table(
    index='ISBN',
    columns='User-ID',
    values='Book-Rating'
).fillna(0)

# Map ISBN → Book Title
isbn_to_title = books.set_index("ISBN")["Book-Title"].to_dict()
title_to_isbn = {v: k for k, v in isbn_to_title.items()}

# -------------------------------------------------
# KNN MODEL
# -------------------------------------------------
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_user_matrix.values)

# -------------------------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------------------------
def get_recommends(book_title):
    if book_title not in title_to_isbn:
        return "Book not found."

    book_isbn = title_to_isbn[book_title]

    if book_isbn not in book_user_matrix.index:
        return "Book not found in filtered dataset."

    book_index = book_user_matrix.index.get_loc(book_isbn)

    distances, indices = model.kneighbors(
        book_user_matrix.iloc[book_index].values.reshape(1, -1),
        n_neighbors=6
    )

    recommendations = []
    for i in range(1, len(distances[0])):
        isbn = book_user_matrix.index[indices[0][i]]
        title = isbn_to_title.get(isbn, "Unknown Title")
        recommendations.append([title, distances[0][i]])

    return [book_title, recommendations]


# -------------------------------------------------
# TEST (FCC EXPECTED OUTPUT FORMAT)
# -------------------------------------------------
result = get_recommends(
    "The Queen of the Damned (Vampire Chronicles (Paperback))"
)

print(result)
