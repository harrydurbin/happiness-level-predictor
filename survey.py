### Harry Durbin
### Happiness Level Survey and Predictor

import numpy as np
import pandas as pd
import sys
import pickle
import random
import warnings

warnings.filterwarnings("ignore")

class Survey:
    def __init__(self):

        self.model = self.loadModel()

        self.health = self.q1()
        self.householdfinances = self.q2()
        self.neighborrace = self.q3()
        self.neighborimmigrant = self.q4()
        self.neighbordiffreligion = self.q5()
        self.neighborunmarriedcpl = self.q6()
        self.lifecontrol = self.q7()
        self.neighborhoodsecurity = self.q8()
        self.countryhumanrights = self.q9()
        self.incomescale = self.q10()
        self.socclass = self.q11()
        self.leisureimport = self.q12()

        self.clf, self.feats = self.compileResponses()
        self.personalhappiness()
        self.prediction = self.predict()

    def loadModel(self):
        # Loading the saved decision tree model pickle
        pkl_filename = 'classifier.pkl'
        model_pkl = open(pkl_filename, 'rb')
        self.model = pickle.load(model_pkl)
        return self.model

    def checkInput(self, var, min, max):
        errmess = "uh-oh, that's not a valid response."
        try:
            int(var)
            if int(var)<=int(max) and int(var)>=min:
                ok = 0
            else:
                ok = 1
                # print '------------------------------------------------------------------'
                print errmess
        except:
            # print '------------------------------------------------------------------'
            print errmess

            ok = 1
        return ok

    def q1(self):
        self.health = raw_input("""All in all, how would you describe your state of health these days? \n
        Would you say it is: \n
        1 : Very Good \n
        2 : Good \n
        3 : Fair \n
        4 : Poor \n""")
        if self.checkInput(self.health, 1, 4) == 1:
            self.q1()
        print '------------------------------------------------------------------'
        return self.health

    def q2(self):
        self.householdfinances = raw_input("""How satisfied are you with the financial situation of your household?: \n
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
        if self.checkInput(self.householdfinances, 1, 10) == 1:
            self.q2()
        print '------------------------------------------------------------------'
        return self.householdfinances

    def q3(self):
        self.neighborrace = raw_input("""Could you please mention if you would not like to have as neighbors?: \n
        ...People of another race \n
        1 : Would not like \n
        2 : Doesn't matter \n""")
        if self.checkInput(self.neighborrace, 1, 2) == 1:
            self.q3()
        print '------------------------------------------------------------------'
        return self.neighborrace

    def q4(self):
        self.neighborimmigrant = raw_input("""
        ...People who are immigrants/foreign: \n
        1 : Would not like \n
        2 : Doesn't matter \n""")
        if self.checkInput(self.neighborimmigrant, 1, 2) == 1:
            self.q4()
        print '------------------------------------------------------------------'
        return self.neighborimmigrant

    def q5(self):
        self.neighbordiffreligion = raw_input("""
        ...People who are are different religion: \n
        1 : Would not like \n
        2 : Doesn't matter \n""")
        if self.checkInput(self.neighbordiffreligion, 1, 2) == 1:
            self.q5()
        print '------------------------------------------------------------------'
        return self.neighbordiffreligion

    def q6(self):
        self.neighborunmarriedcpl = raw_input("""
        ...People who are an unmarried couple living together: \n
        1 : Would not like \n
        2 : Doesn't matter \n""")
        if self.checkInput(self.neighborunmarriedcpl, 1, 2) == 1:
            self.q6()
        print '------------------------------------------------------------------'
        return self.neighborunmarriedcpl

    def q7(self):
        self.lifecontrol = raw_input("""Some people feel they have completely free choice and control over their lives, while other people feel that what they do has no real effect on what happens to them. Please indicate how much freedom of choice and control you feel you have over the way your life turns out.
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
        if self.checkInput(self.lifecontrol, 1, 10) == 1:
            self.q7()
        print '------------------------------------------------------------------'
        return self.lifecontrol

    def q8(self):
        self.neighborhoodsecurity = raw_input("""Could you tell me how secure do you feel these days in your neighborhood?: \n
        1 : Very secure \n
        2 : Quite secure \n
        3 : Not very secure \n
        4 : Not at all secure \n """)
        if self.checkInput(self.neighborhoodsecurity, 1, 4) == 1:
            self.q8()
        print '------------------------------------------------------------------'
        return self.neighborhoodsecurity

    def q9(self):
        self.countryhumanrights = raw_input("""How much respect is there for individual human rights nowadays in this country? Do you feel there is: \n
        1 : A great deal of respect for individual human rights \n
        2 : Fairly much respect \n
        3 : Not much respect \n
        4 : No respect at all \n """)
        if self.checkInput(self.countryhumanrights, 1, 4) == 1:
            self.q9()
        print '------------------------------------------------------------------'
        return self.countryhumanrights

    def q10(self):
        self.incomescale = raw_input("""An income scale on which 1 indicates the lowest income group and 10 the highest income group in your country. Please specify the appropriate number, counting all incomes that come in.
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
        if self.checkInput(self.incomescale, 1, 10) == 1:
            self.q10()
        print '------------------------------------------------------------------'
        return self.incomescale

    def q11(self):
        self.socclass = raw_input("""People sometimes describe themselves as belonging to the working class, the middle class, or the upper or lower class. Would you describe yourself as belonging to the: \n
        1 : Upper class \n
        2 : Upper middle class \n
        3 : Lower middle class \n
        4 : Working class \n
        5 : Lower class \n""")
        if self.checkInput(self.socclass, 1, 4) == 1:
            self.q11()
        print '------------------------------------------------------------------'
        return self.socclass

    def q12(self):
        self.leisureimport = raw_input("""Indicate how important leisure time is in your life: \n
        1 : Very important \n
        2 : Rather important \n
        3 : Not very important \n
        4 : Not at all important \n""")
        if self.checkInput(self.leisureimport, 1, 4) == 1:
            self.q12()
        print '------------------------------------------------------------------'
        return self.leisureimport

    def personalhappiness(self):
        print '#################################################################'
        hap = raw_input("""Do not answer, but just try to think about how happy you feel on a scale of 1-4, and hit enter when done.\n
        1 : VERY HAPPY \n
        2 : RATHER HAPPY \n
        3 : NOT VERY HAPPY \n
        4 : NOT AT ALL HAPPY \n""")
        print '------------------------------------------------------------------'
        print 'Please wait a sec while I load the model and make predictions...'

        quotes = ["'The very purpose of our life is to seek happiness.'\n--the Dalai Lama",
                "'Happiness depends on ourselves.'\n--Aristotle",
                "'For every minute you are angry you lose sixty seconds of happiness.'\n--Ralph Waldo Emerson",
                "'Folks are usually about as happy as they make their minds up to be.'\n--Abraham Lincoln",
                "'Happiness is when what you think, what you say, and what you do are in harmony.'\n--Mahatma Gandhi"]

        i = random.randint(0,len(quotes)-1)
        print '###################################################################'
        print quotes[i]
        return

    def compileResponses(self):
        self.clf = self.loadModel()

        feats = [self.health, self.householdfinances, self.neighborrace, self.neighborimmigrant,
        self.leisureimport, self.lifecontrol, self.neighborhoodsecurity, self.countryhumanrights,
        self.incomescale, self.socclass, self.neighbordiffreligion, self.neighborunmarriedcpl]

        feats_int = map(int, feats)
        feats = np.asarray(feats_int)
        self.feats = feats.reshape(1,-1)

        return self.clf, self.feats

    def predict(self):
        self.prediction = self.clf.predict(self.feats)
        return self.prediction

if __name__ == "__main__":
    print '##################################################################'
    print '##################################################################'
    answers = Survey()

    pred_dict = {1: 'VERY HAPPY -- I hope it is correct.',\
    2: 'RATHER HAPPY -- I hope you are even happier than my prediction.',\
    3: 'NOT VERY HAPPY -- I hope my prediction is wrong (I make mistakes) and you actually feel happier.', # Situations can sometimes suck. If you feel you need more peace, maybe you could try daily meditation.',\
    4: 'NOT AT ALL HAPPY -- Uh-oh, I hope my prediction is wrong (I make mistakes) and you actually feel happier.'} # . Situations can sometimes suck. If you feel you need more peace, maybe you could try daily meditation.'}

    print '###################################################################'
    print """Based on similar survey responses, I predict you are: {}""".format(pred_dict[answers.prediction[0]])
    print '###################################################################'
