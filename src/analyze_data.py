
import utilities
import data_preprocessing
from optparse import OptionGroup, OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
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

def FitModels (models, productCode, cutOff, kFold=10, doPlot=True):
  """Fit an arbitrary number of models to the data and evaluate their
     performance in a boxplot.

  Keyword Arguments:
  models - List of models to fit to the data.
  productCode - products to evaluate the models on
  cutOff - the point at which we determine someone to be branded or non-branded
  kFold - number of folds to use for cross validation strategy
  doPlot - whether we do the boxplot or not
  """
  (X, Y) = data_preprocessing.PrepareData (productCode, cutOff)

  names = []
  results = []

  for name, model in models:
    scores = cross_val_score (model, X, Y.ratio.ravel (), cv=kFold, scoring='accuracy')
    results.append (scores)
    names.append (name)
    msg = "%s: %f (%f)" % (name, scores.mean(), scores.std())
    print (msg)

  if doPlot:
    fig = plt.figure ()
    fig.suptitle('Algorithm Comparison')
    plt.boxplot (results)
    x = fig.add_subplot (111)
    x.set_xticklabels (names)
    plt.savefig (utilities.outputFolder + str (productCode) + ' Algorithm Comparison')

def main ():
  parser = OptionParser ()

  generalGroup = OptionGroup (parser, "General Options", "These options may be requirements for any other options.") 
  generalGroup.add_option ('-p', '--product-code', dest='product_code', type='string')
  generalGroup.add_option ('-c', '--cut-off', dest='cutoff', type=float)

  modelGroup = OptionGroup (parser, "Model Options", "These options control what models to run on the dataset.")
  modelGroup.add_option ('--run-svm', action='store_true', dest='run_svm', default=False)
  modelGroup.add_option ('--run-logistic', action='store_true', dest='run_logistic', default=False)
  modelGroup.add_option ('--run-ada-boost', action='store_true', dest='run_ada_boost', default=False)

  utilityGroup = OptionGroup (parser, "Utility Options", "These options are meant to provide extra functionality" \
                              " such as creating histograms of the brand purchases for each household.")
  utilityGroup.add_option ('--create-histograms', action='store_true', dest='create_histograms', default=False)
  utilityGroup.add_option ('--create-single-histogram', action='store_true', dest='create_single_histogram', default=False)

  parser.add_option_group (generalGroup)
  parser.add_option_group (modelGroup)
  parser.add_option_group (utilityGroup)
  
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

    models = []

    if options.run_svm == True:
      models.append (('SVM', SVC ()))

    if options.run_logistic == True:
      models.append (('Logistic Regression', LogisticRegression ()))

    if options.run_ada_boost == True:
      models.append (('AdaBoost', AdaBoostClassifier (n_estimators=200)))

    if len (models) > 0:
      FitModels (models, options.product_code, options.cutoff, 10, True)

if __name__ == "__main__":
  main ()
