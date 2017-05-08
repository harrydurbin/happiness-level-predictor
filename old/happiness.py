# coding: utf-8

### Harry Durbin
### December 2015
### Happiness Level Predictor

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cluster import MiniBatchKMeans
import sys

print '------------------------------------------------------------------'
algorithm = raw_input('Enter the type of classifier would you like to use \n \
(use "rf" for random forest, "dt" for decision tree, \n \
or "nn" for nearest neighbor):\n ')
algorithm_dict = {'rf': 'random forest', 'dt': 'decision tree', 'nn': 'nearest neighbor'}
print '------------------------------------------------------------------'
try:
    algorithm_dict[algorithm]
except:
    print "That wasn't a valid response. Goodbye."
    sys.exit()
print 'Ok, a {} model...'.format(algorithm_dict[algorithm])

class HappinessPredictor:

    def __init__(self,fn):
        self.raw_data = self.getSurveyData(fn) # load raw survey data
        self.X, self.lat, self.lng = self.getCoords() # get centroid for each country
        self.mbkm_labels = self.clusterCountries() # clusters countries that are near each other
        self.concat_data = self.concatRawData() # joins the cluster labels to the raw survey data
        self.clean_raw = self.cleanData() # cleans errant survey responses
        self.target, self.features = self.extractFeatures() # extracts key features from the raw data
        self.features_train, self.features_test, self.target_train, self.target_test = self.splitData() # create train/test sets
        self.clf = self.createModel(type =algorithm) # creates a model to the train data
        self.predictions = self.makePredictions() # predicts target happiness level using model on test data
        self.accuracy_score, self.cv_score  = self.getCV() # cross validates predictions with actual happiness levels

    def getSurveyData(self,fn):
        # data set from World Value Survey
        print '------------------------------------------------------------------'
        print 'Loading survey data...'
        fn = 'WV6.csv' # approximately 85,000 individual survey results take in 60 countries
        df_raw = pd.read_csv(fn, low_memory = False)
        self.raw_data = df_raw.values
        print '...There are {} rows of data.'.format(len(self.raw_data))
        return self.raw_data

    def getCoords(self):
        # data set for country codes and coordinates
        print '------------------------------------------------------------------'
        print 'Loading country coordinates...'
        fn2 = 'Country_List_ISO_3166_Codes_Latitude_Longitude.csv'
        name_i = [] # country name index
        abbr_i = []  # country abbreviation
        code_i = [] # country code
        lat_i = []  # approximate latitude of country center
        lng_i = []  # approximate longitude of country center
        with open(fn2) as f:
            for line in f: # go over country data, line by line
                x = line.split(',') # get a list of attributes, as strings
                name_i.append(x[0]) # country name index
                abbr_i.append(x[2]) # country abbreviation
                code_i.append(int(x[3])) # country code
                lat_i.append(float(x[4])) # approximate latitude of country center
                lng_i.append(float(x[5])) # approximate longitiude of country center
        lat = [] # latitudes for all points in complete raw survey file
        lng = [] # longitudes for all points in complete raw survey file
        abbr = [] # abbreviations for all points in complete raw survey file
        name = [] # country names for all points in complete raw survey file
        for y in range(len(self.raw_data)):
            for z in range(len(name_i)):
                if int(self.raw_data[y,1]) == int(code_i[z]):
                    lat.append(lat_i[z])
                    lng.append(lng_i[z])
                    abbr.append(abbr_i[z])
                    name.append(name_i[z])
        self.lat = np.asarray(lat)
        self.lng = np.asarray(lng)
        self.name = name
        self.X = np.vstack((lat,lng)).T # array of country coordinates for all of the raw data
        return self.X, self.lat, self.lng

    def clusterCountries(self):
        # initializing minibatch kmeans
        print '------------------------------------------------------------------'
        print 'Creating country clusters...'
        n = 10
        mbkm = MiniBatchKMeans(n_clusters=n, init='k-means++', max_iter=100, batch_size=100,
                         verbose=0, compute_labels=True, random_state=None, tol=0.0,
                               max_no_improvement=10, init_size=None, n_init=3,
                               reassignment_ratio=0.01)
        # running minibatch kmeans
        mbkm.fit(self.X)
        mbkm_labels = mbkm.labels_ # this is an array that indicates cluster
        mbkm_labels = np.asarray(mbkm_labels)
        mbkm_cluster_centers = mbkm.cluster_centers_
        mbkm_labels_unique = np.unique(mbkm_labels)
        self.mbkm_labels = mbkm_labels
        return self.mbkm_labels

    def concatRawData(self):
        print '------------------------------------------------------------------'
        print 'Adding columns to dataset (lat, long, and cluster group)...'
        # create array of latitude, longitude, and cluster group
        raw_add_on = np.vstack((self.lat.T,self.lng.T,self.mbkm_labels.T))
        # combines the raw data with the new lat, lng, cluster array
        self.concat_data = np.vstack((self.raw_data.T,raw_add_on)).T
        return self.concat_data

    def cleanData(self):
        print '------------------------------------------------------------------'
        print 'Cleaning data, removing errant and missing survey responses...'
        # delete row from array if values are missing/inaccurate in the survey
        # column numbers are referenced from features list below
        raw = self.concat_data
        raw = raw[~(raw[:,10]<1)]  # V10: would you say you are 1 V. Happy, 2 Rather Happy, # 3 Not v. happy, 4 not at all happy
        raw = raw[~(raw[:,11]<1)]  # V11: how would you describe your state of health these days? # 1 V. Good, 2 Good, 3 Fair, 4 Poor
        raw = raw[~(raw[:,59]<1)] # V57: Are you: 1 Married, 2 Living together,3 Divorced, # 4 Separated, 5 Widowed, 6 Single
        raw = raw[~(raw[:,60]<0)] # V58: Have you had any children? 0 - 8
        raw = raw[~(raw[:,163]<1)] # V143: do you think about the meaning and purpose of life? # 1 Often, 2 sometimes, 3 rarely, 4 never
        raw = raw[~(raw[:,166]<1)] # V146: how often do you pray? 1 sev time per day, 2 once/day # 3 sev times per week, 4 only services, 5 holy days, 6 1/yr, 7 less, 8 never
        raw = raw[~(raw[:,167]<1)] # V147: would you say you are: 1 religious, 2 not religious, 3 athiest
        raw = raw[~(raw[:,297]<1)] # V229: are you employed now? hours/wk? YES: 1 full time, 2 partime, 3 self, # NO: 4 retired, 5 housewife, 6 student, 7 unempl., 8 other
        raw = raw[~(raw[:,299]<1)] # V231: are work tasks manual or intellectual? 1=mostly manual, 10=mostly intellectual
        raw = raw[~(raw[:,300]<1)] # V232: are work tasks routine or creative? 1=mostly routine, 10=mostly creative
        raw = raw[~(raw[:,306]<1)] # V238: would you describe yourself in: 1 upper class, 2 upper mid class, # 3 low mid, 4 working, 5 lower
        raw = raw[~(raw[:,308]<1)] # V240: gender, 1=male, 2=female
        raw = raw[~(raw[:,310]<1)] # V242: how old are you? 00-99
        raw = raw[~(raw[:,318]<1)] # V248: highest educational level attained? # 1-no formal education... 9-university level w/ degree
        raw = raw[~(raw[:,324]<1)] # V253: Size of town: 1-under 2,000 ... 8-500,000 and more
        self.clean_raw = raw
        return self.clean_raw

    def extractFeatures(self):
        print '------------------------------------------------------------------'
        print 'Extracting important features from raw data...'
        ## This is a list of the important features to extract from raw data
        # the code V### after the feature name is the variable code number
        data = self.clean_raw
        happy = data[:,10]              # Col 10 - Happiness (1-4), V10
        country = data[:,1]             # Col 2 - Country Code, V2A
        marital = data[:,59]            # Col 59 - Marital Status, V57
        kids = data[:,60]               # Col 60 - No. of Children, V58
        rel = data[:,167]               # Col 167 - Religious, V147
        scl = data[:,306]               # Col 306 - Social Class, V238
        income = data[:,307]            # Col 307 - Scale of Income, V239
        age = data[:,310]               # Col 310 - Age, V242
        town = data[:,324]              # Col 324 - Size of Town, V253
        health = data[:,11]             # Col 11 - Health, V11
        purpose = data[:,163]           # Col 163 - Thoughts about life meaning, V143 ###
        employment = data[:,297]        # Col 297 - Employment Status, V229
        intellect_work = data[:,299]    # Col 299 - Manual v Intellectual Work, V231 ###
        creative_work = data[:,300]     # Col 300 - Routine v Creative Work, V232 ####
        sex = data[:,308]               # Col 308 - Sex, V240
        education = data[:,318]         # Col 318 - Education, V248
        pray = data[:,166]              # col 166 - Prayer, V146
        latitude = data[:,430]
        longitude = data[:,431]
        cluster = data[:,432]
        target = happy
        features = np.vstack((country, marital, kids, rel, scl, income, age, town, health,
                          purpose, employment, intellect_work,creative_work,sex,education,
                          pray, latitude, longitude, cluster))
        self.target = target
        self.features = features.T
        return self.target, self.features

    def splitData(self):
        print '------------------------------------------------------------------'
        print 'Splitting data into test and train sets...'
        kf = cross_validation.KFold(n=len(self.target), n_folds=10,
                                    shuffle=True, random_state=None)
        for train_index, test_index in kf:
            self.features_train, self.features_test = self.features[train_index], self.features[test_index]
            self.target_train, self.target_test = self.target[train_index], self.target[test_index]
        return self.features_train, self.features_test, self.target_train, self.target_test

    def createModel(self,type='rf'):
        print '------------------------------------------------------------------'
        print 'Training a predictive model...'
        if type == 'rf':
            model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10)
        elif type == 'dt':
            model =  DecisionTreeClassifier(min_samples_split=20, random_state=99)
        elif type == 'nn':
            model = KNeighborsClassifier(n_neighbors=10)
        else:
            print 'Error: No model type selected!'
            sys.exit()

        self.clf = model.fit(self.features_train, list(self.target_train))
        return self.clf

    def makePredictions(self):
        print '------------------------------------------------------------------'
        print 'Making predictions...'
        self.predictions = self.clf.predict(self.features_test)
        return self.predictions

    def getCV(self):
        print '------------------------------------------------------------------'
        print 'Cross-validating the predictions with training data targets...'
        cv = [] # RMSE
        correct = 0
        for row in range(len(self.predictions)):
            cv.append((((self.predictions[row]) - self.target_test[row])**2)**0.5)
            if self.predictions[row] == self.target_test[row]:
                correct += 1
        self.cv_score = (100-round(np.mean(cv)*100/np.mean(self.target_test)))/100
        self.accuracy_score = round(correct*100/len(self.predictions),2)
        print 'When using the {} algorithm:'.format(algorithm_dict[algorithm])
        print 'The RMSE is {}.'.format(self.cv_score)
        print 'The prediction accuracy is {} %.'.format(self.accuracy_score)
        return self.accuracy_score,self.cv_score

if __name__ == "__main__":
    fn = 'WV6.csv'
    HappinessPredictor(fn)
    print '------------------------------------------------------------------'
    print 'Terminating...'
    sys.exit()
