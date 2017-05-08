### Harry Durbin
### December 2015
### Happiness Level Predictor

import numpy as np
import pandas as pd
import sys
import pickle

# class Survey:
#
#     def __init__(self):
#         self

def loadModel():
    # Loading the saved decision tree model pickle
    pkl_filename = 'classifier.pkl'
    model_pkl = open(pkl_filename, 'rb')
    model = pickle.load(model_pkl)
    return model

def survey():

    health = raw_input("""All in all, how would you describe your state of health these days? Would you say it is: \n
                                Very Good: 1 \n Good: 2 \n
                                Fair: 3 \n Poor: 4 \n """)
    print '------------------------------------------------------------------'

    householdfinances = raw_input("""How satisfied are you with the financial situation of your household?: \n
                                Completely disatisfied: 1 \n 2, 3, 4, 5, 6, 7, 8, 9 \n
                                Completely satisfied: 10 \n """)
    print '------------------------------------------------------------------'

    neighborrace = raw_input("""Could you please mention if you would not like to have as neighbors?: \n
                                ...People of another race \n
                                Would not like: 1 \n Doesn't matter: 2 \n """)
    print '------------------------------------------------------------------'

    neighborimmigrant = raw_input("""Could you please mention if you would not like to have as neighbors?: \n
                                ...People who are immigrants/foreign \n
                                Would not like: 1 \n Doesn't matter: 2 \n """)
    print '------------------------------------------------------------------'

    neighbordiffreligion = raw_input("""Could you please mention if you would not like to have as neighbors?: \n
                                ...People who are are different religion \n
                                Would not like: 1 \n Doesn't matter: 2 \n """)
    print '------------------------------------------------------------------'

    neighborunmarriedcpl = raw_input("""Could you please mention if you would not like to have as neighbors?: \n
                                ...People who are an unmarried couple living together \n
                                Would not like: 1 \n Doesn't matter: 2 \n """)
    print '------------------------------------------------------------------'

    lifecontrol = raw_input("""Some people feel they have completely free choice and control over their lives, while other people
                            feel that what they do has no real effect on what happens to them. Please use this scale where 1
                            means "no choice at all" and 10 means "a great deal of choice" to indicate how much freedom of
                            choice and control you feel you have over the way your life turns out \n
                                    No choice at all: 1 \n 2, 3, 4, 5, 6, 7, 8, 9 \n
                                    A great deal of choice: 10 \n """)
    print '------------------------------------------------------------------'

    neighborhoodsecurity = raw_input("""Could you tell me how secure do you feel these days in your neighborhood?: \n
                                Very secure: 1 \n Quite secure: 2 \n
                                Not very secure: 3 \n Not at all secure: 4 \n """)
    print '------------------------------------------------------------------'

    countryhumanrights = raw_input("""How much respect is there for individual human rights nowadays \
                                in this country? Do you feel there is: \n
                                A great deal of respect for individual human rights: 1 \n
                                Fairly much respect: 2 \n
                                Not much respect: 3 \n
                                No respect at all: 4 \n """)
    print '------------------------------------------------------------------'

    incomescale = raw_input("""On this card is an income scale on which 1 indicates the lowest income group and 10 the highest
                                income group in your country. We would like to know in what group your household is. Please
                                specify the appropriate number, counting all wages, salaries, pensions and other incomes that come
                                in. \n
                                Lowest group: 1 \n 2, 3, 4, 5, 6, 7, 8, 9 \n
                                Highest group: 10 \n """)
    print '------------------------------------------------------------------'

    socclass = raw_input("""People sometimes describe themselves as belonging to the working class, the middle class, or the
                        upper or lower class. Would you describe yourself as belonging to the: \n
                                Upper class: 1 \n
                                Upper middle class: 2 \n
                                Lower middle class: 3 \n
                                Working class: 4 \n
                                Lower class: 5 \n """)
    print '------------------------------------------------------------------'

    leisureimport = raw_input("""Indicate how important leisure time is in your life: \n
                                Very important: 1 \n Rather important: 2 \n
                                Not very important: 3 \n Not at all important: 4 \n """)
    print '------------------------------------------------------------------'

    feats = [health,householdfinances, neighborrace, neighborimmigrant, leisureimport, \
            lifecontrol, neighborhoodsecurity, countryhumanrights, incomescale, socclass, \
            neighbordiffreligion, neighborunmarriedcpl]

    feats_int = map(int,feats)
    feats = np.asarray(feats_int)
    feats = feats.reshape(1,-1)
    return feats

def predict(clf,features):
    print '------------------------------------------------------------------'
    print 'Making prediction about your happiness level...'
    prediction = clf.predict(features)
    return prediction

if __name__ == "__main__":
    feats = survey()
    clf = loadModel()
    prediction = predict(clf,feats)
    pred_dict = {1: 'Very Happy', 2: 'Rather Happy', 3: 'Not very happy' , 4: 'Not at all happy'}
    print 'Disclaimer: Of course this may not be correct, but based on other peoples responses, \n it is predicted you have a happiness level of: {}.'.format(pred_dict[prediction[0]])
    print '------------------------------------------------------------------'

    # sys.exit()
