#importing necessary libraries.
import pandas as pd
import numpy as np

# import TfidfVector from sklearn.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

employees = pd.read_csv(r"D:\UNIVERSITY\PEOPLE ANALYTICS\PA_Code\data\fau_onboarding.csv")
print(employees.columns)

def create_soup(x):
    return ''.join(x['department']) + ''.join(x['published_topics']) + '' + ''.join(x['liked_articles']) + '' + ''.join(x['engagement'])
employees['soup'] = employees.apply(create_soup, axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(employees['soup'])
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# construct a reverse map of indices and employee IDs
indices = pd.Series(employees.index, index=employees['id']).drop_duplicates()

def get_recommendations(ID, cosine_sim=cosine_sim):
    
    # get the index of the employee that matches the employee ID
    IDx = indices[ID]
    
    # get the pairwise similarity scores of all employees with the specified employee ID
    sim_scores = list(enumerate(cosine_sim[IDx]))
    
    # sort employees based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # get the scores of the three most similar employees
    sim_scores = sim_scores[1:4]
    
    # get employee indices
    employees_indices = [i[0] for i in sim_scores]
    
    # return the top three most similar employees
    return employees['id'].iloc[employees_indices]

print(get_recommendations('emp_030', cosine_sim))