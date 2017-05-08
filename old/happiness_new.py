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

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from geonamescache import GeonamesCache
from mpl_toolkits.basemap import Basemap

# This is a list of the important features to extract from raw data
# the code V### after the feature name is the variable code number
#
#     Happiness (1-4), V10
#     Country Code, V2A
#     Marital Status, V57
#     No. of Children, V58
#     Religious, V147
#     Social Class, V238
#     Scale of Income, V239
#     Age, V242
#     Size of Town, V253
#     Health, V11
#     Thoughts about life meaning, V143
#     Employment Status, V229
#     Manual v Intellectual Work, V231
#     Routine v Creative Work, V232
#     Sex, V240
#     Education, V248
#     Prayer, V146

class HappinessPredictor:

    def __init__(self,fn):
        self.raw_data, self.col_names = self.getSurveyData(fn) # load raw survey data
        self.X = self.getCoords() # get centroid for each country
        self.clean_raw = self.cleanData() # cleans errant survey responses
        self.train, self.test = self.splitData() # create train/test sets
        self.features_train, self.features_test, self.target_train, self.target_test = self.extractFeatures() # extracts key features from the raw data
        self.clf = self.createModel() # creates a model to the train data
        self.predictions, self.df_predictions = self.makePredictions() # predicts target happiness level using model on test data
        self.accuracy_score, self.cv_score  = self.getCV() # cross validates predictions with actual happiness levels


    def getSurveyData(self,fn):
        # data set from World Value Survey
        print '------------------------------------------------------------------'
        print 'Loading survey data...'
        use_cols = ['V10','V2A','V57','V58','V147','V238','V239','V242','V253','V11','V143','V229',
                 'V231','V232','V240','V248','V146']
        self.col_names = ['happiness','cntrycode','marriage','children','religious','socclass','income','age',
                      'town','health','thoughts','employment','intellectualwork','creativework','sex',
                      'education','prayer']
        df_raw = pd.read_csv(r"data\WV6.csv", low_memory = False)
        df_trimmed = df_raw[use_cols]
        df_trimmed.columns = self.col_names
        self.raw_data = df_trimmed
        print '...There are {} rows of data.'.format(len(self.raw_data))
        return self.raw_data, self.col_names

    def getCoords(self):
        # data set for country codes and coordinates
        print '------------------------------------------------------------------'
        print 'Appending data with additional information...'
        raw_data = self.raw_data
        df_countries = pd.read_csv(r"data\country_list.csv", header=None)
        df_countries.columns = ['countryname','abbr2','abbr3','cntrycode','lat','lng']
        df_merged = raw_data.merge(df_countries, on = ['cntrycode'])
        self.X = df_merged
        return self.X

    def cleanData(self):
        print '------------------------------------------------------------------'
        print 'Removing errant and missing survey responses...'
        # delete row from array if values are missing/inaccurate in the survey
        # column numbers are referenced from features list below
        raw = self.X
        for col in self.columns:
            raw_temp = raw[raw[col] > 0]
            raw = raw_temp
        self.clean_raw = raw
        print '...There are {} rows of data.'.format(len(self.clean_raw))
        return self.clean_raw

    def splitData(self):
        print '------------------------------------------------------------------'
        print 'Splitting data into test and train sets...'
        kf = cross_validation.KFold(n=len(self.clean_raw), n_folds=2,
                                    shuffle=True, random_state=None)
        for train_index, test_index in kf:
            self.train, self.test = self.clean_raw.ix[train_index,:], self.clean_raw.ix[test_index,:]
        print '...There are {} rows of {} data.'.format(self.train.shape[0], 'train')
        print '...There are {} rows of {} data.'.format(self.test.shape[0], 'test')
        return self.train, self.test

    def extractFeatures(self):
        print '------------------------------------------------------------------'
        print 'Separating the features and target...'
        ## This is a list of the important features to extract from raw data
        target_col = 'happiness'
        feature_cols = ['marriage','children','religious','socclass','income','age',
                      'town','health','thoughts','employment','intellectualwork',
                      'creativework','sex', 'education','prayer']
        self.target_train = self.train[target_col]
        self.target_test = self.test[target_col]
        self.features_train = self.train[feature_cols]
        self.features_test = self.test[feature_cols]
        print self.features_train.ix[0,:]
        print self.target_train.ix[0]
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
        print self.features_train.as_matrix()[1,:]
        print self.target_train.values[1]
        # self.clf = model.fit(self.features_train.as_matrix(), self.target_train.values)
        return #self.clf

    def makePredictions(self):
        print '------------------------------------------------------------------'
        print 'Making predictions...'
        self.predictions = self.clf.predict(self.features_test)
        df_test = self.test
        df_test['predictions'] = self.predictions
        self.df_predictions = df_test
        return self.predictions, self.df_predictions

    def getCV(self):
        print '------------------------------------------------------------------'
        print 'Cross-validating the predictions with training data targets...'
        cv = []
        correct = 0
        for row in range(len(self.predictions)):
            cv.append((((self.predictions[row]) - self.target_test[row])**2)**0.5)
            if self.predictions[row] == self.target_test[row]:
                correct += 1
        self.cv_score = (100-round(np.mean(cv)*100/np.mean(self.target_test)))/100
        self.accuracy_score = round(correct*100/len(self.predictions),2)
        # print 'When using the {} algorithm:'.format(algorithm_dict[algorithm])
        print 'The RMSE is {}.'.format(self.cv_score)
        print 'The prediction accuracy is {} %.'.format(self.accuracy_score)
        return self.accuracy_score,self.cv_score

    def groupCountries(self,dataframe):
        print '------------------------------------------------------------------'
        print 'Grouping surveys into country groups...'
        avg_data = dataframe.groupby('abbr3')['happiness'].mean().sort_values()
        df_avg_data = pd.DataFrame(avg_data)
        df_avg_data = df_avg_data.ix[iso3_codes].dropna()
        df_avg_data = df_avg_data[df_avg_data.index != 'EGY']
        return df_avg_data

    def makeMap(self, dataframe):
        values = dataframe['happiness']
        bmap = brewer2mpl.get_map('YlOrRd', 'sequential', 9)
        cm = bmap.get_mpl_colormap(N=1000, gamma=2.0)
        scheme = [cm(i / float(num_colors)) for i in range(num_colors)]
        bins = np.linspace(values.min(), values.max(), num_colors)
        df_avg_data_woEGY['bin'] = np.digitize(values, bins) - 1
        mpl.style.use('seaborn-white')
        fig = plt.figure(figsize=(22, 12))
        year = '2016'
        ax = fig.add_subplot(111, axisbg='w', frame_on=False)
        fig.suptitle('Happiness Levels {}'.format(year), fontsize=30, y=.95)
        m = Basemap(lon_0=0, projection='robin')
        m.drawmapboundary(color='w')
        m.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
        for info, shape in zip(m.units_info, m.units):
            iso3 = info['ADM0_A3']
            if iso3 not in dataframe.index:
                # adding unique case for egypt
                if iso3 == 'EGY':
                    color = scheme[8]
                else:
                    color = '#dddddd'
            else:
                color = scheme[int(dataframe.ix[iso3]['bin'])]
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches)
            pc.set_facecolor(color)
            ax.add_collection(pc)
        # Cover up Antarctica so legend can be placed over it.
        ax.axhspan(0, 1000 * 1800, facecolor='w', edgecolor='w', zorder=2)
        # Draw color legend.
        ax_legend = fig.add_axes([0.35, 0.14, 0.3, 0.03], zorder=3)
        cmap = mpl.colors.ListedColormap(scheme)
        cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
        cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])
        # Set the map footer.
        plt.annotate(descripton, xy=(-.8, -3.2), size=14, xycoords='axes fraction')
        plt.savefig('happiness2.png', bbox_inches='tight', pad_inches=.2)
        return

if __name__ == "__main__":
    fn = 'WV6.csv'
    HappinessPredictor(fn)
    print '------------------------------------------------------------------'
    makeMap(groupCountries(self.test))
    print 'Terminating...'
    sys.exit()
