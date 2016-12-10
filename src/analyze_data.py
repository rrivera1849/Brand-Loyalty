
import utilities
import data_preprocessing
from optparse import OptionGroup, OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.cluster import KMeans

def CreateAllProductHistograms ():
  """Creates every product ratio histogram.
  """
  productCodes = analyze_data.productHierarchy.product_module_code
  productCodes = productCodes.unique ().tolist ()

  for pc in productCodes:
    if pc >= 0:
      print ('Processing Product Code %d' % pc)
      data_preprocessing.GetHouseholdProductFrequencyByCode (int (pc), True)

def Cluster (description, numClusters=3, isProductGroup=False):
  (X, Y) = data_preprocessing.GetProductFeatures (description, isProductGroup)

  kMeans = KMeans (n_clusters=numClusters)
  kMeans.fit (X)
  labels = kMeans.labels_
  
  figure =  plt.figure (1, figsize=(4, 3))

  ax = Axes3D(figure, rect=[0, 0, .95, 1], elev=48, azim=134)
  ax.scatter(X.iloc[:, 2], X.iloc[:, 0], X.iloc[:, 1], c=labels.astype(np.float))

  ax.w_xaxis.set_ticklabels([])
  ax.w_yaxis.set_ticklabels([])
  ax.w_zaxis.set_ticklabels([])
  ax.set_xlabel('Brand Price Ratio')
  ax.set_ylabel('Brand vs Non-Brand Ratio')
  ax.set_zlabel('Unit Price')

  plt.savefig (utilities.outputFolder + description + '-' + str (numClusters) + '-cluster')
  plt.show ()
  plt.close (figure)

  print 'Clusters saved to output folder.'

def FitModels (models, productCode, cutOff, kFold, featureEvaluation, doPlot):
  """Fit an arbitrary number of models to the data and evaluate their
     performance in a boxplot.

  Keyword Arguments:
  models - List of models to fit to the data.
  productCode - products to evaluate the models on
  cutOff - the point at which we determine someone to be branded or non-branded
  kFold - number of folds to use for cross validation strategy
  featureEvaluation - if true, we will remove features recursively and evaluate which
                      are the best
  doPlot - whether we do the boxplot or not
  """
  (X, Y) = data_preprocessing.PrepareData (productCode, cutOff)
  columnNames = X.columns.values.tolist ()

  names = []
  results = []

  for name, model in models:
    names.append (name)
    scores = None

    if featureEvaluation:
      rfecv = RFECV (estimator=model, step=1, cv=kFold, scoring='accuracy')
      rfecv.fit (X, Y.ratio.ravel ())
      scores = rfecv.grid_scores_

      print "Optimal Number of Features:"
      print rfecv.n_features_

      print "Feature Ranking:"
      featureRanking = []
      for f in range (0, len (rfecv.ranking_)):
        featureRanking.append ( (columnNames[f], rfecv.ranking_[f]) )

      featureRanking.sort (key=lambda x: x[1])

      featureRankingFile = open (utilities.outputFolder + str (productCode) + "-" + \
                                 name + "-" + "feature-ranking", "w+")
      for f in range (0, len (featureRanking)):
        print "\t%d - %s\n" % (featureRanking[f][1], featureRanking[f][0])
        print >> featureRankingFile, "\t%d - %s\n" % (featureRanking[f][1], featureRanking[f][0])
      featureRankingFile.close ();

      if doPlot:
        fig = plt.figure ()
        plt.xlabel ("Number of Features Selected")
        plt.ylabel ("CV Score")
        plt.plot (range (1, len (rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.savefig (utilities.outputFolder + str (productCode) + '-' + \
                     name + '-' + 'feature-evaluation')
        plt.close (fig)

    else:
      scores = cross_val_score (model, X, Y.ratio.ravel (), cv=kFold, scoring='accuracy')

    results.append (scores)

    msg = "%s: %f (%f)" % (name, scores.mean(), scores.std())
    print (msg)

  if doPlot:
    fig = plt.figure ()
    fig.suptitle('Algorithm Comparison Product Code %s' % productCode)
    plt.boxplot (results)
    x = fig.add_subplot (111)
    x.set_xticklabels (names)
    plt.savefig (utilities.outputFolder + str (productCode) + '-algorithm-comparison')
    plt.close (fig)

def main ():
  parser = OptionParser ()

  #=========================General Group Start=========================#
  generalGroup = OptionGroup (parser, "General Options", "These options may be requirements for any other options.") 
  generalGroup.add_option ('-p', '--product-code', dest='product_code', type='string')
  generalGroup.add_option ('-c', '--cut-off', dest='cutoff', type=float, default=0.5)
  #=========================General Group End=========================#

  #=========================Model Group Start=========================#
  modelGroup = OptionGroup (parser, "Model Options", "These options control what models to run on the dataset.")

  modelGroup.add_option ('-k', '--k-fold', dest='k_fold', type=int, default=10)
  modelGroup.add_option ('--do-plots', dest='do_plots', action='store_true', default=False)
  modelGroup.add_option ('--run-svm', action='store_true', \
                         dest='run_svm', default=False)

  modelGroup.add_option ('--run-logistic', action='store_true', \
                         dest='run_logistic', default=False)

  modelGroup.add_option ('--run-ada-boost', action='store_true', \
                         dest='run_ada_boost', default=False)

  modelGroup.add_option ('--recursive-feature-evaluation', action='store_true', \
                         dest='recursive_feature_evaluation', default=False)
  #=========================Model Group End=========================#

  #=========================Utility Group Start=========================#
  utilityGroup = OptionGroup (parser, "Utility Options", \
                              "These options are meant to provide extra functionality " \
                              "such as creating histograms of the brand purchases for each household.")

  utilityGroup.add_option ('--create-histograms', action='store_true', \
                           dest='create_histograms', default=False)

  utilityGroup.add_option ('--create-single-histogram', action='store_true', \
                           dest='create_single_histogram', default=False)

  utilityGroup.add_option ('--cluster', action='store_true', \
                           dest='cluster', default=False)

  utilityGroup.add_option ('--num-clusters', dest='num_clusters', type=int, default=3)

  utilityGroup.add_option ('--department-description', type='string', \
                           dest='department_description')

  utilityGroup.add_option ('--product-group-description', type='string', \
                           dest='product_group_description')
  #=========================Utility Group End=========================#

  parser.add_option_group (generalGroup)
  parser.add_option_group (modelGroup)
  parser.add_option_group (utilityGroup)
  
  (options, args) = parser.parse_args ()

  utilities.ReadAllPickledObjects ()

  if options.create_histograms:
    CreateAllProductHistograms ()

  if options.cluster:
    if options.product_group_description:
      Cluster (options.product_group_description, options.num_clusters, True)
    elif options.department_description:
      Cluster (options.department_description, options.num_clusters, False)
    else:
      parser.error ("Must pass in department description or product group description in order to cluster by department")

  runModel = options.run_svm | options.run_logistic | options.run_ada_boost

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
      FitModels (models, options.product_code, options.cutoff, options.k_fold, \
                 options.recursive_feature_evaluation, options.do_plots)

if __name__ == "__main__":
  main ()
