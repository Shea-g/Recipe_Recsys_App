import pickle

"""

UTILS 

Helper functions to use in recommender.py and app.py

i) Model:
    - nmf_model: trained sklearn NMF model, using user mean imputation

ii) Data: imported from pickled files, in the form of pickled data frames
    -pickled R array with the averages per col
    -pickled model
    -pickled top 100 dict for top 100 recipes currently
    -pickled dictionary with key value pairs of recipes names, and 
    recipe names, ingredients, and instructions


"""

'''
Pickled objects for use 
in recommender 
and app, contained in the 
main directory
'''

#load NMF model, get components (Q)
with open(r"nmf_130.pickle", "rb") as input_file:
    model = pickle.load(input_file)
    Q = model.components_ #also poss to put Q as pickle?? probably..

# Load pickled files: 
# Top 100
with open(r"top100_dict.pickle", "rb") as input_file:
    top100_dict = pickle.load(input_file)


# Dictionary with recipe names (keys) and recipe components and instructions (values)
with open(r"recipes_dict.pickle", "rb") as input_file:
    recipes_dict = pickle.load(input_file) 

# Pickled R col average vector (to use in imputation part of recommender..)
with open(r"R_vec.pickle", "rb") as input_file:
    R_vec = pickle.load(input_file) #need to call it in NMF function! 
