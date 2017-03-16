
from happiness import Happiness

Happiness().getSurveyData() # loads the raw survey data
Happiness().getCoords() # matches up the country centroid lat and long
Happiness().clusterCountries() # clusters countries that are near each other
Happiness().concatRawData() # joins the cluster labels to the raw survey data
Happiness().cleanData() # cleans errant survey responses
Happiness().extractFeatures(): # extracts key features from the raw data
Happiness().splitData() # splits data into train and test sets
Happiness().createModel('rf') # creates a model to the train data
Happiness().makePredictions() # predicts target happiness level using model on test data
Happiness.getCV() # cross validates predictions with actual happiness levels on test data
