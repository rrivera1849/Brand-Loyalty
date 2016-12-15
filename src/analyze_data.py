
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
from sklearn.feature_selection import RFECV

def CreateAllProductHistograms ():
  """Creates every product ratio histogram.
  """
  productCodes = analyze_data.productHierarchy.product_module_code
  productCodes = productCodes.unique ().tolist ()

  for pc in productCodes:
    if pc >= 0:
      print ('Processing Product Code %d' % pc)
      data_preprocessing.GetHouseholdProductFrequencyByCode (int (pc), True)

def Cluster (description, numClusters, isProductGroup, doPlot):
  """Given a department description, or a product group description. This 
     function cluster's similar groups together and returns a dictionary of 
     ProductModuleCode -> cluster

  Keyword Arguments:
  description - a string representing either a department description or a group description
  numClusters - total number of clusters to use for KMeans
  isProductgroup - true if description string is a product group description, false if it's department
  doPlot - if true, we do a 3D plot of our clusters and save them to a file
  """
  productCodesToClusters = data_preprocessing.ClusterInternal (description, isProductGroup, numClusters, doPlot)

  productToClustersFile = open (utilities.outputFolder + description + "-" + \
                                str (numClusters), "w+")

  for key, value in productCodesToClusters.iteritems ():
    print "\t%d - %d\n" % (int (key), int (value))
    print >> productToClustersFile, "\t%d - %d\n" % (int (key), int (value))

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
  names = []
  results = []

  for name, model in models:
    print 'Preparing Data!'
    (X, Y) = data_preprocessing.PrepareData (productCode, cutOff)
    columnNames = X.columns.values.tolist ()

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
      print 'Running Model'
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

def ClusterPicture ():
  productCode='3625,1344,1346,1111,1109,3577,7012,1100,1193,7290,7295,7301,1188,8420,8425,8403,1060,4000,2672,8404,1033,5000'
  productCode = productCode.split (',')

  productDescriptions = []
  for pc in productCode:
    productDescriptions.append \
        (utilities.productHierarchy[utilities.productHierarchy['product_module_code'] == int (pc)].department_descr.to_string (index=False))

  productDescriptions = list (set (productDescriptions))

  numClusters = 2
  productsToClusters = ClusterInternal ("::".join (productDescriptions), False, numClusters, True)

def ClusterAnalysis ():
  productCode='3625,1344,1346,1111,1109,3577,7012,1100,1193,7290,7295,7301,1188,8420,8425,8403,1060,4000,2672,8404,1033,5000'

  results = []
  numClusterList = range (1, 13)

  print 'Preparing X Y Data'
  (X, Y) = data_preprocessing.PrepareDataNoCluster (productCode, cutOff=0.5)
  X['cluster'] = np.ones (X.shape[0]) * 0

  print 'Creating Product Features'
  productCode = productCode.split (',')
  productDescriptions = []
  for pc in productCode:
    productDescriptions.append \
        (utilities.productHierarchy[utilities.productHierarchy['product_module_code'] == int (pc)].department_descr.to_string (index=False))
  productDescriptions = list (set (productDescriptions))

  description = "::".join (productDescriptions)

  if '::' in description:
    description = description.split ('::')
    XProd = pd.DataFrame ()
    YProd = pd.Series ()
  else:
    (XProd, YProd) = GetProductFeatures (description, isProductGroup)

  if type (description) is list:
    for descr in description:
      (xNew, yNew) = GetProductFeatures (descr, False)
      XProd = pd.concat ([XProd, xNew])
      YProd = pd.concat ([YProd, yNew])

  for numCluster in numClusterList:
    print 'Preparing cluster feature'

    kMeans = KMeans (n_clusters=numCluster)
    kMeans.fit (XProd)
    labels = kMeans.labels_
    productsToClusters = dict (zip (YProd.tolist (), labels.tolist ()))

    uniqueProductCodes = X.product_code.unique ().tolist ()
    for pc in uniqueProductCodes:
      print pc
      # X[X['product_code'] == int (pc)].cluster = productsToClusters[int (pc)]
      X.loc[X.product_code == int (pc), 'cluster'] = productsToClusters[int (pc)]

    print X.cluster.value_counts ()

    columnNames = X.columns.values.tolist ()
    # scaler = StandardScaler ()
    # X = pd.DataFrame (scaler.fit_transform (X))
    X.columns = columnNames

    print 'Running logistic regression'
    clf = LogisticRegression ()
    scores = cross_val_score (clf, X.drop ('product_code', 1), Y.ratio.ravel (), cv=10, scoring='accuracy')
    results.append (scores)

    msg = "NumClusters=%d :: %f\n" % (numCluster, scores.mean ())
    print msg

  print numClusterList
  print results

  fig = plt.figure ()
  plt.title ('KNN Performance')
  plt.ylabel ('KFold Cross Validation Accuracy K=%d' % 10)
  plt.xlabel ('Number of Clusters')
  plt.gca ().set_xlim ([0, 13])
  plt.plot (numClusterList, results, 'b')
  plt.plot (numClusterList, results, 'bo')
  plt.savefig (utilities.outputFolder + str (productCode) + '-cluster-analysis')
  plt.close (fig)

  print 'Cluster Analysis Finished!'

def main ():
  parser = OptionParser ()

  #=========================General Group Start=========================#
  generalGroup = OptionGroup (parser, "General Options", "These options may be requirements for any other options.") 
  generalGroup.add_option ('-p', '--product-code', dest='product_code', type='string')
  generalGroup.add_option ('-c', '--cut-off', dest='cutoff', type=float, default=0.5)
  generalGroup.add_option ('--no-multinomial-prediction', dest='multinomial_prediction', \
                           action='store_false', default=True)
  generalGroup.add_option ('--no-cluster-feature', dest='cluster_feature', \
                           action='store_false', default=True)
  generalGroup.add_option ('--use-product-feature', dest='product_feature', \
                           action='store_true', default=False)
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

  data_preprocessing.MULTINOMIAL_PREDICTION = options.multinomial_prediction
  data_preprocessing.CLUSTER_FEATURE = options.cluster_feature
  data_preprocessing.PRODUCT_FEATURE = options.PRODUCT_FEATURE

  utilities.ReadAllPickledObjects ()

  if options.create_histograms:
    CreateAllProductHistograms ()

  if options.cluster:
    if options.product_group_description:
      Cluster (options.product_group_description, options.num_clusters, True, options.do_plots)
    elif options.department_description:
      Cluster (options.department_description, options.num_clusters, False, options.do_plots)
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
      models.append (('SVM Binary', SVC ()))

    if options.run_logistic == True:
      models.append (('Logistic Regression Binary', LogisticRegression ()))
      models.append (('Logistic Regression Multinomial', LogisticRegression ()))

    if options.run_ada_boost == True:
      models.append (('AdaBoost Binary', AdaBoostClassifier (n_estimators=200)))

    if len (models) > 0:
      FitModels (models, options.product_code, options.cutoff, options.k_fold, \
                 options.recursive_feature_evaluation, options.do_plots)

if __name__ == "__main__":
  main ()
