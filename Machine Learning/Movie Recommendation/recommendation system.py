#!/usr/bin/env python
# coding: utf-8

# ## RECOMMENDATION SYSTEM
# - Movie Recommender System
# - Content based filtering
# - - In this file, we worked on data cleaning for recommendations system and tokenization

# ### TOPICS WE HAVE DISCUSSED (16 august class):
# - Filtering = collaborative and content based filtering
# - chicken and egg problem in business
# - Vectors
# - Similarity
# - vectorisation
# - bag of words
# - stemming
# - cosine similarity
# - list comprehension

# In[1]:


import pandas as pd
import json


# In[2]:


movies = pd.read_csv('../Datasets/movies.csv')
credits = pd.read_csv('../Datasets/credits.csv')


# In[3]:


movies.shape, credits.shape


# In[4]:


json.loads(movies["genres"][0])[0]


# In[5]:


movies.columns


# In[6]:


credits.columns


# - Reason for using JSON:
#   - The 'genres' column stores data as a STRING that looks like a list of dictionaries.
#   - To actually work with it as a real Python dictionary, we need the JSON library.

# In[7]:


# Without JSON: # Returns a plain STRING — we cannot access keys or loop through it

movies["genres"][0]


# In[8]:


# With JSON: 
    # json.loads() converts the STRING into a real Python LIST of DICTIONARIES
    # Now we can access keys like ["name"], loop through it, and extract values
json.loads(movies["genres"][0])


# In[9]:


# We set index to = 0 to get first line
json.loads(movies["genres"][0])[0]

# At index = 0, we write ['name'] to retrieve genre name. 
    #It's like going in without loop
json.loads(movies["genres"][0])[0]['name']


# In[10]:


## DATA MERGING

movies_credits = pd.merge(movies, credits, left_on="id", right_on="movie_id")


# In[11]:


movies_credits.shape


# In[12]:


movies_credits.columns


# In[13]:


movies_credits = movies_credits[["movie_id","title_x","overview","genres","keywords",'cast', 'crew']]
movies_credits


# In[14]:


movies_credits.dropna(inplace=True)


# In[15]:


values = json.loads(movies["genres"][0])
values

# Using List Comprehension in Python
[value["name"] for value in values]


# - To understand above List Comprehension. We actually did for loop to save 'name' of 'genres'
# - Following Code is lenghty version of above list comprehension ↓

# In[16]:


# list_of_genres_names = []

# for value in values:
#     list_of_genres_names.append(value["name"])

# list_of_genres_names


# In[17]:


def extract_values(str_list):
    values = json.loads(str_list)
    return [value["name"] for value in values]


# In[18]:


extract_values(movies["genres"][0])


# In[19]:


# Applied extract_values Function to dataFrame, now we get list of genres
movies_credits["genres"] = movies_credits["genres"].apply(extract_values)

# Same goes for keywords Column
movies_credits["keywords"] = movies_credits["keywords"].apply(extract_values)


# In[20]:


# Now we want to do same for 'cast column' but we only want top 3 cast names not extra
movies_credits["cast"].apply(extract_values) # this will return every name


# In[21]:


movies_credits["cast"] = movies_credits["cast"].apply(extract_values).apply(lambda y:y[:3]) # this will return only 3 initial cast name
movies_credits["cast"]


# - Now we are going to fetch director

# In[22]:


# Function to fetch Director:
def fetch_director(str_list):
    values = json.loads(str_list)
    return [value["name"] for value in values if value["job"] == "Director"]


# In[23]:


fetch_director(movies_credits["crew"][0])


# In[24]:


# # Convert the 'crew' column string into a real Python list of dictionaries
# values = json.loads(movie_credits["crew"][0])

# # Loop through every crew member in the list
# for value in values:
#     job  = value["job"]   # extract the job title of this crew member
#     name = value["name"]  # extract the name of this crew member

#     # We only want the Director — filter out everyone else
#     if job == "Director":
#         print(name, " : ", job)


# In[25]:


movies_credits["crew"] = movies_credits["crew"].apply(fetch_director)
movies_credits


# In[26]:


movies_credits["overview"] = movies_credits["overview"].str.split()


# In[27]:


def collapse(lst):
    final_lst = []
    for i in lst:
        final_lst.append(i.replace(" ", ""))
    return final_lst


# In[28]:


collapse(["Action","Adventure","Fantasy","Science Fiction"])


# In[29]:


movies_credits["cast"] = movies_credits["cast"].apply(collapse)
movies_credits["crew"] = movies_credits["crew"].apply(collapse)
movies_credits["genres"] = movies_credits["genres"].apply(collapse)
movies_credits["keywords"] = movies_credits["keywords"].apply(collapse)


# In[30]:


movies_credits["tags"] = movies_credits["overview"] + movies_credits["genres"] + movies_credits["keywords"] + movies_credits["cast"]


# In[31]:


final_df = movies_credits[["movie_id", "title_x", "tags"]]


# **Class 23/8/26**
# - Discussion on Explainatory vs Exploratory Data Analysis
# - In Interview, Always answer in a way so no question from question comes
# - bag of words, tf idf
# - sklearn and CountVectorizer
# - stemming
# - vectors and its comparison
# - cosine similarity

# In[32]:


# Following work is required for machine learning model.
# ML model does not need tokens or list, it needs paragraph so we are converting it from list to string of paragraph

# ml model function input requires-> str -> split() 


# In[33]:


# This is the behind the scene work of below code
" ".join(final_df["tags"][0])


# In[34]:


final_df["tags"] = final_df["tags"].apply(lambda x: " ".join(x)).str.lower()


# In[35]:


final_df["tags"][0]


# ### Now We will do stemming

# In[36]:


# NLP helps in machine understanding human language
import nltk


# In[37]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[38]:


ps.stem("loved")


# In[39]:


ps.stem("loving")


# In[40]:


ps.stem("dancing")


# In[41]:


ps.stem("Danced")


# In[42]:


# Function for stemming using for loops
def stem(txt):
    lst = []

    for i in txt.split():
        lst.append(ps.stem(i))

    return " ".join(lst)


# In[43]:


final_df["tags"] = final_df["tags"].apply(stem)


# In[44]:


final_df.head()


# In[45]:


## sklearn and CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[46]:


cv = CountVectorizer(max_features=5000, stop_words="english")


# In[47]:


vector = cv.fit_transform(final_df["tags"])


# In[48]:


vector = vector.toarray()


# In[49]:


# features names
list(cv.get_feature_names_out())


# In[50]:


final_df.shape


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity


# In[52]:


# Vectors comparison with cosine similarity

similarity = cosine_similarity(vector)
similarity.shape


# In[53]:


final_df.reset_index(drop=True, inplace=True)
final_df


# In[54]:


## Recommend 
## Recommendation while watching movie


final_df[final_df["title_x"] == "Titanic"]


# In[55]:


# Now I want vector of 'Titanic' movie
# so we KNOW 'Titanic' movie is at '25' vector so we will simply do following step to get vector of 'titanic' movie

vector[25]


# In[56]:


# we get index number of titanic movie
final_df[final_df["title_x"] == "The Dark Knight Rises"].index


# In[57]:


# we only wanted titanic movie index number so we provided index[0]
# You can compare with above code output to better understand the idea
movie_index = final_df[final_df["title_x"] == "Avatar"].index[0]
movie_index


# In[58]:


# we will save similarity of 'titanic' movie in distances. Folling is the similarity score of 'titanic' with other vectors.
distances = similarity[movie_index]
distances


# In[59]:


# By applying enumarte function, it assigned each value index number
index_list = list(enumerate(distances))
index_list


# - With this, we have synced index number in following variables 'distances, final_df, vector'

# In[60]:


# So now we wanted top 10 similar movies
# we will sort, then get 10 similar movies
# Sort function with 'reverse=True' will sort values in descending order on basis of 'index_list'. we use'key= lambda x:x[1]' to achieve this

similar_movie = sorted(index_list, reverse=True, key= lambda x:x[1])
similar_movie


# In[61]:


# we get second number of movie using loc function
final_df.loc[1599]


# In[62]:


for i in similar_movie:
    print(i)


# In[63]:


# We wanted top 10 movies to be recommended similar to 'titanic' movie:

for index, similarity_score in similar_movie[1:11]: #[1:11] means top 10 movies

    print(final_df.iloc[index, 1])


# In[64]:


# Now we are saving this whole recommendation system in a single function

def recommend(movie_name):
    movie_index = final_df[final_df["title_x"] == movie_name].index[0]
    distances = similarity[movie_index]
    similar_movie = sorted(index_list, reverse=True, key= lambda x:x[1])

    print(f"Top 10 Movies similar to {movie_name}\n")
    for index, similarity_score in similar_movie[1:11]: #[1:11] means top 10 movies

        print(final_df.iloc[index, 1])


# In[65]:


recommend("The Dark Knight Rises")

