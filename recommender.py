"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""
import numpy as np
import pandas as pd
from utils import recipes_dict, model, Q, R_vec

#Sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF


def NMF_new_user(user_rating, recipes_dict, R_vec):  #user rating zipped dict in app file
    
    """
    Creates a df with the user input.
    Takes the pickled dictionary index and the 
    user input to create a user df. Cols iterated
    through to remove duplicates before concat. 
    User df then imputed with user-wise and 
    recipe wise means, and averaged. 
    Future: impute with cat wise weighted mean
    """
    visitor ='new_user'
    user_input = pd.DataFrame(user_rating, index=[visitor])
    
    #create necessary df
    recipe =pd.DataFrame(columns=recipes_dict.keys()) #takes values from pickled dict and makes into col names

    #fix the issue of repeat cols
    recipe_cols = list(recipe.columns)
    user_input_cols = list(user_input.columns)
    #print(user_input_cols)
                           
    for col in recipe_cols:
        if col in  user_input_cols:
            recipe.drop([col], axis=1, inplace=True)
        else: 
            pass
    new_user = pd.concat([recipe, user_input]) 

     #impute user with values (steps below)
    user1 = new_user.T.fillna(new_user.mean(axis=1).round(2)).T.values
    user2 = R_vec #R_vec is one row of the col wise imputed df as an array
    user_i = (user1 + user2)/2
    user_i
    return(user_i) 




def NMF_model(user_i,model, Q, recipes_dict): 

    '''
    takes imputer cold-start user vector 
    and transforms it using the model 
    to output a list of recommended recipe names

    Future: hybrid model and NLP (with ratings)
    '''

    user_P = model.transform(user_i)
    user_R = np.dot( user_P, Q)
    
    #df of predictions
    reco = pd.DataFrame({'user_imputed':user_i[0], 'predicted_ratings':user_R[0]}, index = recipes_dict.keys())
    reco_10 = reco.sort_values(by ='predicted_ratings', ascending= False).iloc[:10,:]
    recommendation =list(reco_10.index)

    return (recommendation) #formatted as list



def NMF_output(user_rating):

    '''
    Small iterative function to go through list 
    of recommendations and return list 
    of matched dict values (with recipe instructions)
    '''
    user_i = NMF_new_user(user_rating, recipes_dict, R_vec)
    recommendation = NMF_model(user_i,model, Q, recipes_dict)

    recoreco = []

    for k in recommendation:
        if k in recipes_dict.keys():
            recoreco.append(recipes_dict[k])

    return(recoreco)