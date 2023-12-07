#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:46:44 2023

@author: tarunvannelli
"""

import streamlit as st
import os
import re
#import nltk
import requests
import warnings
import pandas as pd
import numpy as np
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt
import re
from itertools import cycle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from PIL import Image
warnings.filterwarnings('ignore')
from dataprep.eda import plot, plot_correlation, plot_missing, create_report


st.title(':blue[_Book Recommendation System_] :books:')

books_df = pd.read_csv("C:\\Users\\avina\\Documents\\Data Science\\Projects\\P-186- Book Recommendation\\final_data.csv",encoding='latin-1')
books_df['title'] = books_df['title'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))


# explicit data
explicit=books_df[books_df['rating']!=0]
explicit=explicit.reset_index(drop=True)
print(explicit.shape)

##Popularity Based
explicit.title.value_counts().reset_index()

ratings = pd.DataFrame(explicit.groupby('title')['rating'].mean())
ratings.rename({'rating':'avg_ratings'}, axis=1 , inplace =True)
ratings['num_ratings'] = pd.DataFrame(explicit.groupby('title')['rating'].count())
ratings.reset_index(inplace=True)

popular_books = ratings[ratings['num_ratings']>=100].sort_values('avg_ratings', ascending=False).head(10)
popular_books.reset_index(inplace=True, drop=True)
popular_books.sort_values('num_ratings', ascending=False)

twentyfive_popular_books = popular_books.merge(explicit, on='title').drop_duplicates('title').reset_index(drop=True)
twentyfive_popular_books = twentyfive_popular_books[['title','avg_ratings','num_ratings','author','publisher','image_url_m']]






recom_option = st.sidebar.selectbox(
    '**:violet[Select the Option Below]**',
    ('Popular Books', 'Recommendation Based on Title', 'Recommendation Based on Author'))




def fun_to_dsiplay_images_content(filteredImages,caption):
    idx = 0 
    for _ in range(len(filteredImages)): 
        cols = st.columns(5) 
        
        if idx < len(filteredImages): 
            cols[0].image(filteredImages[idx], width=120, caption=caption[idx])
        idx+=1
        
        if idx < len(filteredImages):
            cols[1].image(filteredImages[idx], width=120, caption=caption[idx])
        idx+=1
    
        if idx < len(filteredImages):
            cols[2].image(filteredImages[idx], width=120, caption=caption[idx])
        idx+=1 
        if idx < len(filteredImages): 
            cols[3].image(filteredImages[idx], width=120, caption=caption[idx])
            idx = idx + 1
        if idx < len(filteredImages): 
            cols[4].image(filteredImages[idx], width=120, caption=caption[idx])
            idx = idx + 1
        else:
            break


if recom_option == 'Popular Books':
    filteredImages = twentyfive_popular_books['image_url_m'].tolist()
    caption = twentyfive_popular_books['title'].tolist()
    st.write("**:red[Check our Popular Books]**")
    fun_to_dsiplay_images_content(filteredImages,caption)


##Preproessing

user_counts = pd.DataFrame(explicit.groupby('userid').count()['rating']).rename({'rating':'no_times_rated'}, axis=1).reset_index()
final_books = pd.merge(explicit, user_counts, on='userid')
final_books.drop(columns = ['location','isbn','country','age','year'],axis=1,inplace = True) #remove useless cols
final_books = final_books[final_books['no_times_rated']>=10].reset_index(drop=True)

##Removing Books with less than 50 number of User Ratings

final_book = final_books[final_books['no_times_rated'] >= 50]
df = final_book.copy()

##Model-based collaborative filtering system

pivot_tables = df.pivot_table(index='title', columns='userid',values='rating').fillna(0)

from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity matrix
cos_sim_matrix = cosine_similarity(pivot_tables.values)
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=pivot_tables.index, columns=pivot_tables.index)

pivot_tables.index = pivot_tables.index.str.strip()
similarity = cosine_similarity(pivot_tables)

# Defining a Function
def recommend(book_name):
    pivot_tables = df.pivot_table(index='title', columns='userid',values='rating').fillna(0)
    index = np.where(pivot_tables.index==book_name)[0][0]
    similarity = cosine_similarity(pivot_tables)
    similar_items = sorted(list(enumerate(similarity[index])),key = lambda x:x[1], reverse=True)[1:20]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = df[df['title'] == pivot_tables.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('title')['title'].values))
        item.extend(list(temp_df.drop_duplicates('title')['author'].values))
        item.extend(list(temp_df.drop_duplicates('title')['image_url_m'].values))
        
        data.append(item)
    return data





if recom_option == 'Recommendation Based on Title':
    st.write("**:red[Recommendations Based on the Book You Select]**")
    option = st.sidebar.selectbox('**:violet[Select or Type the Title]**',df['title'].tolist())
    
    if option != 0:
        recommend_books = pd.DataFrame(recommend(option),columns=['title', 'author','image']).head(10)
        recomend_imgs = recommend_books['image'].tolist()
        recomend_caption = recommend_books['title'].tolist()
        recomend_author = recommend_books['author'].tolist()
        fun_to_dsiplay_images_content(recomend_imgs, recomend_caption)


#### Author Based Filterings
def author_based_recommender(author_name):
    author_books = df[df['author'] == author_name]
    author_books = author_books.drop_duplicates(subset=["title", "publisher"], keep='first')
    sorted_books = author_books.sort_values(by='rating',ascending=False)
    top_books = sorted_books.head(10)
    return top_books


if recom_option == 'Recommendation Based on Author':
    st.write("**:red[Recommendations Based on the Author You Select]**")
    option = st.sidebar.selectbox('**:violet[Select or Type the Author]**',df['author'].tolist())
    
    if option != 0:
        st.write("Books By Author:",option)
        author_recommend_books = author_based_recommender(option)
        author_recomend_imgs = author_recommend_books['image_url_m'].tolist()
        author_recomend_caption = author_recommend_books['title'].tolist()
        fun_to_dsiplay_images_content(author_recomend_imgs, author_recomend_caption)










