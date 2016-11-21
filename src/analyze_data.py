
import utilities
import data_preprocessing
from optparse import OptionGroup, OptionParser

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def CreateAllProductHistograms ():
  """Creates every product ratio histogram.
  """
  productCodes = analyze_data.productHierarchy.product_module_code
  productCodes = productCodes.unique ().tolist ()

  for pc in productCodes:
    if pc >= 0:
      print ('Processing Product Code %d' % pc)
      data_preprocessing.GetHouseholdProductFrequencyByCode (int (pc), True)


def FitLogisticRegression (productCode, cutOff):
  """Fits a logistic regression model and then evaluates how well
     we did using k-fold validation.

  Keyword Arguments:
  productCode - string or integer representing product codes to use
  cutOff - what brand non-brand cutoff to use when creating our labels
  """
  print ('Running Logistic Regression')
  (X, Y) = data_preprocessing.PrepareData (productCode, cutOff)

  logreg = LogisticRegression ()
  scores = cross_val_score (logreg, X, Y.ratio.ravel (), cv=10)

  return scores

def FitSVM (productCode, cutOff, kernel='rbf'):
  """Fits a support vector machine model and then evaluates how well
     we did using k-fold validation.

  Keyword Arguments:
  productCode - string or integer representing product codes to use
  cutOff - what brand non-brand cutoff to use when creating our labels
  """
  print ('Runnning SVM')
  (X, Y) = data_preprocessing.PrepareData (productCode, cutOff)

  clf = SVC (kernel=kernel)
  scores = cross_val_score (clf, X, Y.ratio.ravel (), cv=10)

  return scores

def FitAdaBoost (productCode, cutOff):
  """Fits a boosting model model and then evaluates how well
     we did using k-fold validation.

  Keyword Arguments:
  productCode - string or integer representing product codes to use
  cutOff - what brand non-brand cutoff to use when creating our labels
  """
  print ('Runnning AdaBoost')
  (X, Y) = data_preprocessing.PrepareData (productCode, cutOff)

  bdt = AdaBoostClassifier (n_estimators=200)
  scores = cross_val_score (bdt, X, Y.ratio.ravel (), cv=10)

  return scores

def FitSVMGrid (productCode, cutOff):
  """Attempts to find the best parameters for a support vector machine model.

  Keyword Arguments:
  productCode - string or integer representing product codes to use
  cutOff - what brand non-brand cutoff to use when creating our labels
  """
  print ('Running SVM Grid...')
  (X, Y) = data_preprocessing.PrepareData (productCode, cutOff)

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
  parser.add_option ('-p', '--product-code', dest='product_code', type='string')
  parser.add_option ('-c', '--cut-off', dest='cutoff', type=float)
  parser.add_option ('--run-svm', action='store_true', dest='run_svm', default=False)
  parser.add_option ('--run-svm-grid', action='store_true', dest='run_svm_grid', default=False)
  parser.add_option ('--run-logistic', action='store_true', dest='run_logistic', default=False)
  parser.add_option ('--run-ada-boost', action='store_true', dest='run_ada_boost', default=False)
  
  (options, args) = parser.parse_args ()

  utilities.ReadAllPickledObjects ()

  if options.create_histograms == True:
    CreateAllProductHistograms ()

  runModel = options.run_svm | options.run_logistic | options.run_svm_grid | options.run_ada_boost

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

    if options.run_ada_boost == True:
      scores = FitAdaBoost (options.product_code, options.cutoff)
      print ("The mean accuracy of ada boost is %0.2f" % scores.mean ())

if __name__ == "__main__":
  main ()
