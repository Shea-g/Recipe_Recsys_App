import re
from flask import Flask, render_template, request
from utils import top100_dict, recipes_dict 
from recommender import NMF_new_user, NMF_model, NMF_output
import random
import pandas as pd
import numpy as np

# construct our flask instance, pass name of module
#requests received from clients passed by server to this object for handling

app = Flask(__name__)

@app.route('/')
def welcome():
    """
    Route for welcome page.
    User can select 3 different recommendation methods.
    """
    # renders the html page as the output of this function
    return render_template ('index.html',name="Variety is the 🌶spice🌶 of life!", recipes= list(recipes_dict.keys()))
    # 'recipes' variable is passed from python file to the html file for accessing it inside the html file - for rating




@app.route('/top_100')
def top_100_recommend():
    """
    Top 100 recipes from the dataset
    Future plans to make top 100 per cat

    """
    recipe_names = list(top100_dict.values())

    return  render_template('top_100.html',recipes_names=recipe_names) 


@app.route('/rand_recommend')

def random_recommend():
    """
    Random recommender, which uses random lib
    and samples the pickled recipes dictionary
    to return 5 random recipes from ~5000

    """
    rand_rec = random.sample(list(recipes_dict.values()), 5) #recommending recipes as defined in pd.read_csv in utils
    # renders the html page as the output of this function

    return  render_template('random_reco.html', rand_rec = rand_rec)




@app.route('/recommender_user')
def nmf_recommend():
    """
    Renders user ratings (using request args)
    feeds the user ratings into the model (in 
    recommender) to get the NMF recommendations

    -> note that user table and imputation 
    performed in model function

    """
    #read user input from url/webpage
    print(request.args)
    recipe = request.args.getlist('recipe') # taking lists of titles only from user input, make sure same in html
    ratings = request.args.getlist('ratings') # taking lists of ratings only from user input, "       "
    print(recipe,ratings)

    # converting lists of titles and ratings into dict to pass to our recommender model
    ratingss = map(int, ratings) #needs to be int
    user_rating = dict(zip(recipe,ratingss)) 

    recoreco = NMF_output(user_rating)
    
    # renders the html page as the output of this function, need to perhaps put all of nmf in one func!
    return  render_template('recommender_user.html', recoreco= recoreco)



if __name__=='__main__':
    app.run(debug=True) #port=5000

