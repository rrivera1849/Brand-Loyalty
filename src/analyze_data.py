
import time 
from optparse import OptionGroup, OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

pickleFolder = 'pickled-objects/'
outputFolder = 'output/'
brandFrequencyFolder = outputFolder + 'product-frequency-by-household/'

def TimeItDecorator (function):
  """Basic decorator that times the runtime of some arbitrary function.
  """
  def Wrapper (*args):
    begin = time.time ()
    ret = function (*args)
    end = time.time ()
    print ('%s took %0.3f ms' % (function.func_name, (end - begin) * 1000.0))
    return ret

  return Wrapper

@TimeItDecorator
def ReadAllPickledObjects ():
  """Reads our previously pickleized dataset
  """
  global brandVariations, products, purchases, trips, panelist, \
         productsExtra, retailers, productHierarchy, dmaToFips

  brandVariations = ReadPickledObject ('brand-variations.pkl')
  products = ReadPickledObject ('products.pkl')
  purchases = ReadPickledObject ('purchases.pkl')
  trips = ReadPickledObject ('trips.pkl')
  panelist = ReadPickledObject ('panelist.pkl')
  productsExtra = ReadPickledObject ('products-extra.pkl')
  retailers = ReadPickledObject ('retailers.pkl')
  productHierarchy = ReadPickledObject ('product-hierarchy.pkl')
  dmaToFips = ReadPickledObject ('dma-to-fips.pkl')

def ReadPickledObject (filename):
  """Read a single pickled object from a filename

  Keyword Arguments:
  filename - the name of the file where our pickled object is stored
  """
  return pd.read_pickle (pickleFolder + filename)

def GetHighestQuantity (k = 10):
  purchases.sort_values ('quantity', inplace=True, ascending=False)
  productsExtra.loc[purchases['upc'][:k].tolist ()].to_csv (outputFolder + 'products-extra-upc-highest-quantity.csv', sep='\t')

def GetUPCFrequency ():
  valueCounts = purchases.upc.value_counts ()
  valueCounts.to_csv (outputFolder + 'purchases-upc-frequency.csv', sep='\t')

def GetHouseholdProductFrequencyByCode (productCode, saveFiles=False):
  productSearch = products[products['product_module_code'] == productCode]

  merged = pd.merge (productSearch, purchases, on=None, left_on=None, right_on='upc', left_index=True, right_index=False)
  merged['trip_code_uc'] = merged.index.tolist ()
  merged['is_brand'] = np.where (merged.brand_descr.str.contains ('CTL') == True, False, True)
  merged = pd.merge (merged, trips, on=None, left_on='trip_code_uc', right_on=None, left_index=False, right_index=True)
  merged = merged.groupby (['household_code', 'is_brand'])['quantity'].sum ().reset_index ()

  merged['brand_purchases'] = np.where (merged.is_brand == True, merged.quantity, 0)
  merged['non_brand_purchases'] = np.where (merged.is_brand == False, merged.quantity, 0)
  merged = merged.groupby (['household_code'])[['brand_purchases', 'non_brand_purchases']].sum ().reset_index ()

  merged['ratio'] = merged.brand_purchases.astype(float) / (merged.brand_purchases.astype(float) + merged.non_brand_purchases.astype(float))

  if saveFiles:
    if len (merged.ratio) > 0:
      merged.to_pickle (brandFrequencyFolder + str (productCode))

    fig = plt.figure ()

    productName = productHierarchy[productHierarchy['product_module_code'] == productCode].product_module_descr.tolist ()[0]
    plt.title (str (productName) + ' - Ratio of Brand Purchases')
    plt.xlabel ('Ratio')
    plt.ylabel ('Frequency')
    merged.ratio.plot.hist ()

    plt.savefig (brandFrequencyFolder + str (productCode) + '.png')
    plt.close (fig)

  return merged

def PrepareData (productCode):
  householdProductFrequency = GetHouseholdProductFrequencyByCode (productCode)

  merged = pd.merge (householdProductFrequency, panelist, on=None, left_on='household_code', right_on=None, left_index=False, right_index=True)
  merged = merged.drop (['panel_year', 'projection_factor', 'projection_factor_magnet', 'household_composition', \
          'male_head_birth', 'female_head_birth', 'marital_status', 'panelist_zip_code', 'fips_state_code', 'fips_state_descr', \
          'fips_county_code', 'fips_county_descr', 'region_code', 'scantrack_market_code', 'scantrack_market_descr', 'dma_code', 'dma_descr', \
          'kitchen_appliances', 'tv_items', 'household_internet_connection', 'wic_indicator_ever_notcurrent', \
          'Member_1_Birth', 'Member_1_Relationship_Sex', 'Member_1_Employment', \
          'Member_2_Birth', 'Member_2_Relationship_Sex', 'Member_2_Employment', \
          'Member_3_Birth', 'Member_3_Relationship_Sex', 'Member_3_Employment', \
          'Member_4_Birth', 'Member_4_Relationship_Sex', 'Member_4_Employment', \
          'Member_5_Birth', 'Member_5_Relationship_Sex', 'Member_5_Employment', \
          'Member_6_Birth', 'Member_6_Relationship_Sex', 'Member_6_Employment', \
          'Member_7_Birth', 'Member_7_Relationship_Sex', 'Member_7_Employment', \
          'type_of_residence', 'male_head_occupation', 'female_head_occupation',\
          'household_code', 'brand_purchases', 'non_brand_purchases' \
          ], 1)

  merged['age_and_presence_of_children'] = np.where (merged.age_and_presence_of_children == 9, 0, 1)
  merged['male_head_employment'] = np.where (merged.male_head_employment.isin ([1, 2, 3]), 1, 0)
  merged['female_head_employment'] = np.where (merged.female_head_employment.isin ([1, 2, 3]), 1, 0)

  merged['race_white'] = np.where (merged.race == 1, 1, 0)
  merged['race_black'] = np.where (merged.race == 2, 1, 0)
  merged['race_asian'] = np.where (merged.race == 3, 1, 0)
  merged['race_other'] = np.where (merged.race == 4, 1, 0)
  merged = merged.drop ('race', 1)

  merged['hispanic_origin'] = np.where (merged.hispanic_origin == 1, 1, 0)
  merged['wic_indicator_current'] = np.where (merged.wic_indicator_current == 1, 1, 0)
  
  labels = pd.DataFrame (merged['ratio'])
  merged = merged.drop ('ratio', 1)

  return (merged, labels)

def CreateAllProductHistograms ():
  productCodes = productHierarchy.product_module_code
  productCodes = productCodes.unique ().tolist ()

  for pc in productCodes:
    if pc >= 0:
      print ('Processing Product Code %d' % pc)
      GetHouseholdProductFrequencyByCode (int (pc), True)

def ModelWrapper (productCode, cutOff):
  (X, Y) = PrepareData (productCode)

  # Scale data
  scaler = StandardScaler ()
  X = pd.DataFrame (scaler.fit_transform (X))

  # Cutoff our ratio
  Y['ratio'] = np.where (Y.ratio > cutOff, 1, 0)

  return (X, Y)

def FitLogisticRegression (productCode, cutOff):
  print ('Running Logistic Regression')
  (X, Y) = ModelWrapper (productCode, cutOff)

  logreg = LogisticRegression ()
  scores = cross_val_score (logreg, X, Y.ratio.ravel (), cv=10)

  return scores

def FitSVM (productCode, cutOff, kernel='rbf'):
  print ('Runnning SVM')
  (X, Y) = ModelWrapper (productCode, cutOff)

  clf = SVC (kernel=kernel)
  scores = cross_val_score (clf, X, Y.ratio.ravel (), cv=10)

  return scores

def FitSVMGrid (productCode, cutOff):
  print ('Running SVM Grid...')
  (X, Y) = ModelWrapper (productCode, cutOff)

  C_range = np.logspace(-3, 3, 5)
  gamma_range = np.logspace(-3, 3, 5)
  param_grid = dict(gamma=gamma_range, C=C_range)

  cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=43)
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
  grid.fit (X, Y.ratio.ravel ())

  return grid

def main ():
  parser = OptionParser ()

  parser.add_option ('--create-histograms', action='store_true', dest='create_histograms', default=False)
  parser.add_option ('--create-single-histogram', action='store_true', dest='create_single_histogram', default=False)
  parser.add_option ('-p', '--product-code', dest='product_code', type=int)
  parser.add_option ('-c', '--cut-off', dest='cutoff', type=float)
  parser.add_option ('--run-svm', action='store_true', dest='run_svm', default=False)
  parser.add_option ('--run-svm-grid', action='store_true', dest='run_svm_grid', default=False)
  parser.add_option ('--run-logistic', action='store_true', dest='run_logistic', default=False)
  
  (options, args) = parser.parse_args ()

  ReadAllPickledObjects ()

  if options.create_histograms == True:
    CreateAllProductHistograms ()

  runModel = options.run_svm | options.run_logistic | options.run_svm_grid

  if runModel or options.create_single_histogram:
    if not options.product_code:
      parser.error ('Must provide a product code for any single product operations')

    if not options.cutoff and runModel:
      parser.error ('Must provide a cutoff for any single product operations')

    if options.create_single_histogram:
      GetHouseholdProductFrequencyByCode (options.product_code, True)

    if options.run_svm == True:
      scores = FitSVM (options.product_code, options.cutoff)
      print ("The mean accuracy of svm is %0.2f" % scores.mean ())

    if options.run_svm_grid == True:
      svmGrid = FitSVMGrid (options.product_code, options.cutoff)
      print("The best parameters are %s with a score of %0.2f" % (svmGrid.best_params_, svmGrid.best_score_))

    if options.run_logistic == True:
      scores = FitLogisticRegression (options.product_code, options.cutoff)
      print ("The mean accuracy of logistic regression is %0.2f" % scores.mean ())

if __name__ == "__main__":
  main ()
