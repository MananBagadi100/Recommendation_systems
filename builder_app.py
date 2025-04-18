import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
#This is the surprise library Randomized Search CV
from surprise.model_selection.search import RandomizedSearchCV 
from surprise import accuracy
from surprise import KNNBasic
from sklearn.ensemble import RandomForestClassifier
from surprise.model_selection import KFold
import pickle
df=pd.read_csv('7817_1.csv')
df.head(3)

print(df['reviews.rating'].value_counts())

df.isnull().sum()

df.shape

#using regex to remove all the special charecters
#lamba performs the respective operation on each row
#apply performs it on the entire column
df['reviews.text']=df['reviews.text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '',x))

#now we are coverting all the reviews into lowercase for easy analysis
# str.lower affects the whole column and is also faster when compared to lambda functions
df['reviews.text']=df['reviews.text'].str.lower()
df['reviews.text'].head(10)

#applying TDF-IF and storing the sparse matrix into a new variable
#by default the TDF-IF is applied on the entire column
#TF-IDF also expects whole sentences or documents as input
vectorizer=TfidfVectorizer()
df_tf_idf_matrix=vectorizer.fit_transform(df['reviews.text'])
print(df_tf_idf_matrix)

#using cosine similarity to find the co-relation between words
cosine_sim_matrix=cosine_similarity(df_tf_idf_matrix)

cosine_sim_matrix

cosine_sim_matrix.shape

# Dropping columns that are not relevant or contain missing values
columns_to_drop = [
    'colors', 'dimension', 'ean', 'manufacturer', 'manufacturerNumber',
    'sizes', 'upc', 'weight', 'reviews.date', 'reviews.doRecommend', 
    'reviews.numHelpful', 'reviews.rating', 'reviews.title', 'reviews.userCity', 
    'reviews.userProvince', 'reviews.username'
]
df_cleaned = df.drop(columns=columns_to_drop)


print(df_cleaned.columns)


print(df_cleaned.isnull().sum())

#here we used the cosine similarity matrix rows as index without having an explicit index in
#original table. this is the reason product ID is able to map to the correct row

#function to get the top recommendations
def get_top_recommendations(product_id, cosine_sim_matrix, N=5):
    similarity_scores=cosine_sim_matrix[product_id]
    #sorting the scores based on descending order
    #remember to even keep track of the indices of the scores to get the similar products
    #np.argsort() returns the indices of the sorted array elements
    similar_indices=np.argsort(similarity_scores)[::-1]
    #removing the product itself
    #it creates a boolean array removing all the false (elements which do not satisfy the condition)
    similar_indices=similar_indices[similar_indices != product_id]
    #get the indices
    top_indices=similar_indices[:N]
    #return the top N recommendations and their similarity scores
    return [(index, similarity_scores[index]) for index in top_indices]


#calling the above function to give recommendations (content recommendations)
#a=int(input('Enter the product ID'))
a=5
n=5
model_recommendations=get_top_recommendations(a-1,cosine_sim_matrix,n)
#printing the recommendation
for idx,score in model_recommendations:
    print(f'Product : {idx} \t Similarity Score : {score}')

#############################
#Collaborative based recommendation system starts here
df.dtypes

#checking the value counts
df['id'].value_counts()

#here we are removing the null value rows which do not have a username
df.dropna(axis=0, subset=['reviews.username'], how='any', inplace=True)

#checking if null values are removed
df['reviews.username'].isnull().sum()

#here factorize function assigns a unique integer to each user 
#multiple rows assigned same id as the same user has done multiple reviews
#here we are assigning one unique integer for each customer
#here we are indexing from 1, so it will be User 1,2,3,4,....
df['user_id']=df['reviews.username'].factorize()[0]+1
print(df[['reviews.username','user_id']].head())


#adding unique product id for each product
df['product_id']=df['asins'].factorize()[0]+1
print(df[['product_id','user_id']].head(5))


df['user_id'].value_counts()

#number of reviews each 
df['product_id'].value_counts()

df

# Finding the total number of unique products
total_products_b = df['product_id'].nunique()
print(f"Total number of unique products: {total_products_b}")

#filtering the products with reviews less than 5 to prevent distortation
min_review=5
#this series contains number of reviews for each product
product_count=df['product_id'].value_counts()
#then it is compared to see it has minimum review threshold and it is stored
filtered_product=product_count[product_count >= min_review].index
#the products which have the review threshold are now compared to the dataframe
#product id's 
df_filtered=df[df['product_id'].isin(filtered_product)]

#above we are removing the  products which have total min reviews as 5

#now we are filtering the users who interacted with min 2 products
#thus further refining our user item matrix
user_counts_1=df_filtered['user_id'].value_counts()
#filter out the user_id which do not satisfy the condition 
#the ones which do we store their indices 
users_to_keep_1=user_counts_1[user_counts_1>=2].index
#compare with all the user_id and remove the ones which do not satisfy the condition
df_filtered_updated=df_filtered[df_filtered['user_id'].isin(users_to_keep_1)]

print(df_filtered_updated.head())
print(df_filtered_updated.shape)

# Finding the total number of unique products after filtering
total_products_after = df_filtered_updated['product_id'].nunique()
print(f"Total number of unique products after filtering: {total_products_after}")

#user item matrix
#it is a pandas dataframe
user_item_matrix=df_filtered_updated.pivot_table(index='user_id',columns='product_id',values='reviews.rating')
user_item_matrix.columns

print(user_item_matrix)

#removing rows where ratings are not given
#This contains only the rows which are rated so that we can train our KNNBasic model on it 
df_given_ratings=df.dropna(subset=['reviews.rating'])


#now we are using another method KNNBasic to predict the missing values
sim_options={
    'name': 'cosine',    #calculates the cosine similarity between all pairs of products
    'user_based': False  #tells to calculate similarites between products
}

reader=Reader(rating_scale=(1,5))
#to load the dataset form dataframe as surprise library cannot read it directly
data=Dataset.load_from_df(df_given_ratings[['user_id','product_id','reviews.rating']],reader)
#splitting the dataset into training and testing sets
train_set,test_set=train_test_split(data,test_size=0.2)
#randomized search will take care of parameters
knnb=KNNBasic(sim_options=sim_options) 

#parameter grid for hypertuning
param_grid={
    'k':[10,20,30,40],
    'k_min':[1,2,3,4]
}

#the algo uses KFolds internally for consistent splits
#applying the randomized search CV' of the surprise library here
random_state=RandomizedSearchCV(algo_class=knnb,param_distributions=param_grid,n_iter=10,measures=['rmse','mae'],cv =3,refit=True,n_jobs=-1,random_state=0)

#training the model
knnb.fit(train_set)
#making predictions
prediction_knnb=knnb.test(test_set)
#measuring the rmse of predictions
accuracy.rmse(prediction_knnb)

user_item_matrix_knnb=user_item_matrix.copy()
#now we will fill the missing values using KNNBasic
for user_id in user_item_matrix_knnb.index:
    for product_id in user_item_matrix_knnb.columns:
        if pd.isna(user_item_matrix_knnb.loc[user_id,product_id]):
            answer_knnb=knnb.predict(user_id,product_id)
            user_item_matrix_knnb.loc[user_id,product_id]=answer_knnb.est


print(user_item_matrix_knnb)

cosine_item_similarity_knnb=cosine_similarity(user_item_matrix_knnb.T)
print(cosine_item_similarity_knnb)

#here what we are basically doing is if a user has given rating on some
#product we recommend similar products to him based on the similarity of 
#the other products which is found by the cosine similarity matrix.
#missing ratings are filled by svd and the input is the product id label
#to my future self to not forget 😅 

# Export cosine similarity matrix
with open('cosine_item_similarity.pkl', 'wb') as file:
    print("🔥 cosine_item_similarity_knnb shape:", cosine_item_similarity_knnb.shape)
    print("🔥 sample row:", cosine_item_similarity_knnb[5][:5])

    pickle.dump(cosine_item_similarity_knnb, file)
# Export user-item matrix
with open('user_item_matrix.pkl', 'wb') as file:
    pickle.dump(user_item_matrix, file)
print('Pickle files saved successfully')