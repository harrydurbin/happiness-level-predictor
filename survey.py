### Harry Durbin
### Happiness Level Survey and Predictor

import numpy as np
import pandas as pd
import sys
import pickle
import random
import warnings

warnings.filterwarnings("ignore")

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
    1 : Very Good \n
    2 : Good \n
    3 : Fair \n
    4 : Poor \n""")
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
    10 : Completely satisfied \n""")
    print '------------------------------------------------------------------'

    neighborrace = raw_input("""Could you please mention if you would not like to have as neighbors?: \n
    ...People of another race \n
    1 : Would not like \n
    2 : Doesn't matter \n""")
    print '------------------------------------------------------------------'

    neighborimmigrant = raw_input("""
    ...People who are immigrants/foreign: \n
    1 : Would not like \n
    2 : Doesn't matter \n""")
    print '------------------------------------------------------------------'

    neighbordiffreligion = raw_input("""
    ...People who are are different religion: \n
    1 : Would not like \n
    2 : Doesn't matter \n""")
    print '------------------------------------------------------------------'

    neighborunmarriedcpl = raw_input("""
    ...People who are an unmarried couple living together: \n
    1 : Would not like \n
    2 : Doesn't matter \n""")
    print '------------------------------------------------------------------'

    lifecontrol = raw_input("""Some people feel they have completely free choice and control over their lives, while other people feel that what they do has no real effect on what happens to them. Please indicate how much freedom of choice and control you feel you have over the way your life turns out.
    1 : No choice at all \n
    2 \n
    3 \n
    4 \n
    5 \n
    6 \n
    7 \n
    8 \n
    9 \n
    10 : A great deal of choice \n""")
    print '------------------------------------------------------------------'

    neighborhoodsecurity = raw_input("""Could you tell me how secure do you feel these days in your neighborhood?: \n
    1 : Very secure \n
    2 : Quite secure \n
    3 : Not very secure \n
    4 : Not at all secure \n """)
    print '------------------------------------------------------------------'

    countryhumanrights = raw_input("""How much respect is there for individual human rights nowadays in this country? Do you feel there is: \n
    1 : A great deal of respect for individual human rights \n
    2 : Fairly much respect \n
    3 : Not much respect \n
    4 : No respect at all \n """)
    print '------------------------------------------------------------------'

    incomescale = raw_input("""An income scale on which 1 indicates the lowest income group and 10 the highest income group in your country. Please specify the appropriate number, counting all incomes that come in.
    1 : Lowest group \n
    2 \n
    3 \n
    4 \n
    5 \n
    6 \n
    7 \n
    8 \n
    9 \n
    10 : Highest group \n""")
    print '------------------------------------------------------------------'

    socclass = raw_input("""People sometimes describe themselves as belonging to the working class, the middle class, or the upper or lower class. Would you describe yourself as belonging to the: \n
    1 : Upper class \n
    2 : Upper middle class \n
    3 : Lower middle class \n
    4 : Working class \n
    5 : Lower class \n""")
    print '------------------------------------------------------------------'

    leisureimport = raw_input("""Indicate how important leisure time is in your life: \n
    1 : Very important \n
    2 : Rather important \n
    3 : Not very important \n
    4 : Not at all important \n""")

    print '#################################################################'
    hap = raw_input("""Do not answer, but just try to think about how happy you feel on a scale of 1-4, and hit enter when done.\n
    1 : VERY HAPPY \n
    2 : RATHER HAPPY \n
    3 : NOT VERY HAPPY \n
    4 : NOT AT ALL HAPPY \n""")

    print 'Please wait a sec while I load the model and make predictions...'

    quotes = ["'The very purpose of our life is to seek happiness.'\n--the Dalai Lama",
            "'Happiness depends on ourselves.'\n--Aristotle",
            "'For every minute you are angry you lose sixty seconds of happiness.'\n--Ralph Waldo Emerson",
            "'Folks are usually about as happy as they make their minds up to be.'\n--Abraham Lincoln",
            "'Happiness is when what you think, what you say, and what you do are in harmony.'\n--Mahatma Gandhi"]

    i = random.randint(0,len(quotes)-1)
    print '###################################################################'
    print quotes[i]

    clf = loadModel()

    feats = [health,householdfinances, neighborrace, neighborimmigrant, leisureimport, \
            lifecontrol, neighborhoodsecurity, countryhumanrights, incomescale, socclass, \
            neighbordiffreligion, neighborunmarriedcpl]

    feats_int = map(int,feats)
    feats = np.asarray(feats_int)
    feats = feats.reshape(1,-1)

    return clf, feats

def predict(clf,features):
    prediction = clf.predict(features)
    return prediction

if __name__ == "__main__":
    clf, feats = survey()
    prediction = predict(clf,feats)

    pred_dict = {1: 'VERY HAPPY -- I hope it is correct.',\
    2: 'RATHER HAPPY -- I hope you are even happier than my prediction.',\
    3: 'NOT VERY HAPPY -- I hope my prediction is wrong (I make mistakes) and you actually feel happier. Situations can sometimes suck. If you feel you need more peace, maybe you could try daily meditation.',\
    4: 'NOT AT ALL HAPPY -- Uh-oh, I hope my prediction is wrong (I make mistakes) and you actually feel happier. . Situations can sometimes suck. If you feel you need more peace, maybe you could try daily meditation.'}

    print '###################################################################'
    print """Based on similar survey responses, I predict you are: {}""".format(pred_dict[prediction[0]]).upper()
    print '###################################################################'
