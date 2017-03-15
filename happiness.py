
# coding: utf-8


import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn import cross_validation
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap


class Happiness():
    
    def __init__(self):
        self
    
    # data set from World Value Survey
    def getSurveyData(self):
        fn = 'WV6.csv' # approximately 85,000 individual survey results take in 60 countries
        raw0 = np.genfromtxt(fn, delimiter=',', skip_header=1)
        Happiness.raw0 = raw0
        
        
    # data set for country codes and coordinates
    def getCoords(self):
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

        for y in range(len(self.raw0)):
            for z in range(len(name_i)):
                if int(raw0[y,1]) == int(code_i[z]):
                    lat.append(lat_i[z]) 
                    lng.append(lng_i[z])
                    abbr.append(abbr_i[z])
                    name.append(name_i[z]) 

        Happiness.lat = np.asarray(lat)
        Happiness.lng = np.asarray(lng)
        Happiness.name = name
        Happiness.X = np.vstack((lat,lng)).T # array of country coordinates for all of the raw data
        
        
    def clusterCountries(self):
    # initializing minibatch kmeans
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

        #compiling countries into 10 cluster groups
        group_a = [], group_b = [], group_c = [], group_d = [], group_e = [], group_f = [], group_g = [], group_h = [],
        group_i = [], group_j = []

        for x in range(len(self.name)):
            for y in range(len(mbkm_labels_unique)):
                if mbkm_labels[x] == 0:
                    group_a.append(str(self.name[x]))
                if mbkm_labels[x] == 1:
                    group_b.append(str(self.name[x]))
                if mbkm_labels[x] == 2:
                    group_c.append(str(self.name[x]))
                if mbkm_labels[x] == 3:
                    group_d.append(str(self.name[x]))
                if mbkm_labels[x] == 4:
                    group_e.append(str(self.name[x]))
                if mbkm_labels[x] == 5:
                    group_f.append(str(self.name[x]))
                if mbkm_labels[x] == 6:
                    group_g.append(str(self.name[x]))
                if mbkm_labels[x] == 7:
                    group_h.append(str(self.name[x]))      
                if mbkm_labels[x] == 8:
                    group_i.append(str(self.name[x]))      
                if mbkm_labels[x] == 9:
                    group_j.append(str(self.name[x]))

        Happiness.mbkm_labels = mbkm_labels

            
    def concatRawData(self):
        # create array of latitude, longitude, and cluster group
        raw1 = np.vstack((self.lat.T,self.lng.T,self.mbkm_labels.T)) 
        # combines the raw data with the new lat, lng, cluster array
        raw = np.vstack((self.raw0.T,raw1)).T 

        Happiness.raw = raw


    def cleanData(self):
    # delete row from array if values are missing/inaccurate in the survey
    # column numbers are referenced from features list below
        raw = self.raw
        # V10: would you say you are 1 V. Happy, 2 Rather Happy, 
        # 3 Not v. happy, 4 not at all happy
        raw = raw[~(raw[:,10]<1)]
        # V11: how would you describe your state of health these days? 
        # 1 V. Good, 2 Good, 3 Fair, 4 Poor
        raw = raw[~(raw[:,11]<1)]
        # V57: Are you: 1 Married, 2 Living together,3 Divorced, 
        # 4 Separated, 5 Widowed, 6 Single
        raw = raw[~(raw[:,59]<1)] 
        # V58: Have you had any children? 0 - 8
        raw = raw[~(raw[:,60]<0)] 
        # V143: do you think about the meaning and purpose of life? 
        # 1 Often, 2 sometimes, 3 rarely, 4 never
        raw = raw[~(raw[:,163]<1)]
        # V146: how often do you pray? 1 sev time per day, 2 once/day 
        # 3 sev times per week, 4 only services, 5 holy days, 6 1/yr, 7 less, 8 never
        raw = raw[~(raw[:,166]<1)]
        # V147: would you say you are: 1 religious, 2 not religious, 3 athiest
        raw = raw[~(raw[:,167]<1)] 
        # V229: are you employed now? hours/wk? YES: 1 full time, 2 partime, 3 self, 
        # NO: 4 retired, 5 housewife, 6 student, 7 unempl., 8 other
        raw = raw[~(raw[:,297]<1)]
        # V231: are work tasks manual or intellectual? 1=mostly manual, 10=mostly intellectual
        raw = raw[~(raw[:,299]<1)]
        # V232: are work tasks routine or creative? 1=mostly routine, 10=mostly creative
        raw = raw[~(raw[:,300]<1)]
        # V238: would you describe yourself in: 1 upper class, 2 upper mid class, 
        # 3 low mid, 4 working, 5 lower
        raw = raw[~(raw[:,306]<1)] 
        # V240: gender, 1=male, 2=female
        raw = raw[~(raw[:,308]<1)]
        # V242: how old are you? 00-99
        raw = raw[~(raw[:,310]<1)] 
        # V248: highest educational level attained? 
        # 1-no formal education... 9-university level w/ degree
        raw = raw[~(raw[:,318]<1)]
        # V253: Size of town: 1-under 2,000 ... 8-500,000 and more
        raw = raw[~(raw[:,324]<1)] 
        
        Happiness.raw = raw

    
    def extractFeatures(self):
        ## This is a list of the important features to extract from raw data
        # the code V### after the feature name is the variable code number

        # Col 10 - Happiness (1-4), V10
        # Col 2 - Country Code, V2A
        # Col 59 - Marital Status, V57
        # Col 60 - No. of Children, V58
        # Col 167 - Religious, V147
        # Col 306 - Social Class, V238
        # Col 307 - Scale of Income, V239
        # Col 310 - Age, V242
        # Col 324 - Size of Town, V253
        # Col 11 - Health, V11
        # Col 163 - Thoughts about life meaning, V143 ###
        # Col 297 - Employment Status, V229
        # Col 299 - Manual v Intellectual Work, V231 ###
        # Col 300 - Routine v Creative Work, V232 ####
        # Col 308 - Sex, V240
        # Col 318 - Education, V248
        # col 166 - Prayer, V146 
        
        raw = self.raw
        
        happy = raw[:,10]
        
        country = raw[:,1]
        marital = raw[:,59]
        kids = raw[:,60]
        rel = raw[:,167]
        scl = raw[:,306]
        income = raw[:,307]
        age = raw[:,310]
        town = raw[:,324]
        health = raw[:,11]
        purpose = raw[:,163]
        employment = raw[:,297]
        intellect_work = raw[:,299]
        creative_work = raw[:,300]
        sex = raw[:,308]
        education = raw[:,318]
        pray = raw[:,166]
        latitude = raw[:,430]
        longitude = raw[:,431]
        cluster = raw[:,432]

        target = happy
        features = np.vstack((country, marital, kids, rel, scl, income, age, town, health,
                          purpose, employment, intellect_work,creative_work,sex,education,
                          pray, latitude, longitude, cluster))
        
        Happiness.target = target
        Happiness.features = features.T

        

    def splitData(self):
        # k-fold data dividing
        kf = cross_validation.KFold(n=len(self.raw), n_folds=10, indices=None, 
                               shuffle=False, random_state=None)
        for train_index, test_index in kf:
            Happiness.features_train, Happiness.features_test = self.features[train_index], self.features[test_index] 
            Happiness.target_train, Happiness.target_test = self.target[train_index], self.target[test_index] 
 

    def createModel(self,type='rf'):
        if type = 'rf':
            model = RandomForestRegressor(random_state=0, n_estimators=100, max_depth=10)
        elif type = 'dt':
            model =  DecisionTreeClassifier(min_samples_split=20, random_state=99)
        elif type = 'nn':
            model =  = KNeighborsRegressor(n_neighbors=10)
        else:
            print 'Error: No model type selected!'
        model = model.fit(Happiness.features_train, Happiness.happy_train) # decision trees
        
        Happiness.model = model
        
        
    def makePredictions(self):
        predictions = Happiness.model.predict(features_test)
        
        Happiness.predictions = predictions
        

    def getCV(self):
        # RMSE for nearest neighbor
        cv = []
        for row in range(len(Happiness.predictions)):
            nn_cv.append(((round(Happiness.prediction[row]) - Happiness.target_test[row])**2)**0.5)
        print np.mean(cv)
        print 100-round(np.mean(cv)*100/np.mean(Happiness.target_test[row])), '%'
        Happiness.cv = np.mean(cv)





