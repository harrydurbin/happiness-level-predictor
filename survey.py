### Harry Durbin
### December 2015
### Happiness Level Predictor

import numpy as np
import pandas as pd
import sys
import pickle

def loadModel():
    # Loading the saved decision tree model pickle
    pkl_filename = 'classifier.pkl'
    model_pkl = open(pkl_filename, 'rb')
    model = pickle.load(model_pkl)
    return model

def survey():
    print '##################################################################'
    print '##################################################################'
    health = raw_input("""All in all, how would you describe your state of health these days? \n
    Would you say it is: \n
    Very Good: 1 \n
    Good: 2 \n
    Fair: 3 \n
    Poor: 4 \n
    \n """)
    print '------------------------------------------------------------------'

    householdfinances = raw_input("""How satisfied are you with the financial situation of your household?: \n
    1 : Completely disatisfied \n
    2 \n
    3 \n
    4 \n
    5 \n
    6 \n
    7 \n
    8 \n
    9 \n
    10 : Completely satisfied \n
     \n """)
    print '------------------------------------------------------------------'

    neighborrace = raw_input("""Could you please mention if you would not like to have as neighbors?: \n
    ...People of another race \n
    Would not like: 1 \n
    Doesn't matter: 2 \n
     \n """)
    print '------------------------------------------------------------------'

    neighborimmigrant = raw_input("""...People who are immigrants/foreign: \n
    Would not like: 1 \n
    Doesn't matter: 2 \n
     \n """)
    print '------------------------------------------------------------------'

    neighbordiffreligion = raw_input("""...People who are are different religion: \n
    Would not like: 1 \n
    Doesn't matter: 2 \n
     \n """)
    print '------------------------------------------------------------------'

    neighborunmarriedcpl = raw_input("""People who are an unmarried couple living together: \n
    Would not like: 1 \n
    Doesn't matter: 2 \n
     \n """)
    print '------------------------------------------------------------------'

    lifecontrol = raw_input("""Some people feel they have completely free choice and control over their\
    lives, while other people feel that what they do has no real effect on what happens to them. Please\
    indicate how much freedom of choice and control you feel you have over the way your life turns out. \n
    1 : No choice at all \
    2 \
    3 \
    4 \
    5 \
    6 \
    7 \
    8 \
    9 \
    10 : A great deal of choice \n
     \n """)
    print '------------------------------------------------------------------'

    neighborhoodsecurity = raw_input("""Could you tell me how secure do you feel these days in your neighborhood?: \n
    Very secure: 1 \n
    Quite secure: 2 \n
    Not very secure: 3 \n
    Not at all secure: 4 \n """)
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
    print '#################################################################'
    print '#################################################################'
    print 'Wait a sec while I load the model...'
    print """Try to think about how happy you feel on a scale of 1-4:\n
    1: VERY HAPPY \n
    2: RATHER HAPPY \n
    3: NOT VERY HAPPY \n
    4: NOT AT ALL HAPPY \n"""

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
    pred_dict = {1: 'VERY HAPPY -- I hope it is correct. You seem to be doing well.',\
    2: 'RATHER HAPPY -- I hope you are even happier than my prediction. You seem to be doing well.',\
    3: 'NOT VERY HAPPY -- I hope my prediction is wrong and you feel happier. Situations can sometimes suck. If you feel you need more peace, maybe you could try daily meditation.',\
    4: 'NOT AT ALL HAPPY -- I hope my prediction is wrong. Situations can sometimes suck. If you feel you need more peace, maybe you could try daily meditation.'}
    print '###################################################################'
    print """Based on similar survey responses, I predict you are:\n
     {}.""".format(pred_dict[prediction[0]])
    print '###################################################################'
