
import itertools

import utilities

import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURE = True
PRODUCT_FEATURE = False
MULTINOMIAL_PREDICTION = True

def GetProductFeatures (description, isProductGroup):
  if isProductGroup:
    search = utilities.productHierarchy[utilities.productHierarchy['product_group_descr'] == description]
  else:
    search = utilities.productHierarchy[utilities.productHierarchy['department_descr'] == description]

  # UPC is the index of the products matrix, we make it a column so we don't
  # lose it in the merge
  toMerge = utilities.products
  toMerge['upc'] = utilities.products.index

  merged = pd.merge (search, toMerge, \
                     on='product_module_code', left_on=None, right_on=None, \
                     left_index=False, right_index=False)

  merged = pd.merge (merged, utilities.purchases, on='upc', left_on=None, right_on=None, \
                     left_index=False, right_index=False)
  
  merged['is_brand'] = np.where (merged.brand_descr.str.contains ('CTL') == True, False, True)
  merged['is_not_brand'] = (merged['is_brand'] - 1) * -1
  merged['brand_price'] = merged['total_price_paid'] * merged['is_brand']
  merged['non_brand_price'] = merged['total_price_paid'] * merged['is_not_brand']
  merged['quantity_brand'] = merged['quantity'] * merged['is_brand']
  merged['quantity_non_brand'] = merged['quantity'] * merged['is_not_brand']

  merged = \
      merged.groupby (['product_module_code'])['total_price_paid', 'quantity', 'is_brand', 'is_not_brand', \
                                               'quantity_brand', 'quantity_non_brand', 'brand_price', 'non_brand_price'].sum ().reset_index ()

  merged['brand_ratio'] = merged['is_brand'] / (merged['is_brand'] + merged['is_not_brand'])
  merged['unit_price'] = merged['total_price_paid'] / merged['quantity']
  merged['brand_price_ratio'] = (merged['brand_price'] / merged['quantity_brand']) / (merged['non_brand_price'] / merged['quantity_non_brand'])

  merged = merged[['brand_ratio', 'unit_price', 'brand_price_ratio', 'quantity', 'product_module_code']]
  merged = merged.dropna ()
  merged = merged[merged['quantity'] >= 1000]

  X = merged[['brand_ratio', 'unit_price', 'brand_price_ratio']]
  Y = merged['product_module_code']

  return (X, Y)

def ClusterInternal (description, isProductGroup, numClusters, doPlot):
  if '::' in description:
    description = description.split ('::')
    X = pd.DataFrame ()
    Y = pd.Series ()
  else:
    (X, Y) = GetProductFeatures (description, isProductGroup)

  if type (description) is list:
    for descr in description:
      (xNew, yNew) = GetProductFeatures (descr, isProductGroup)
      X = pd.concat ([X, xNew])
      Y = pd.concat ([Y, yNew])

  kMeans = KMeans (n_clusters=numClusters)
  kMeans.fit (X)
  labels = kMeans.labels_

  productCodesToClusters = dict (zip (Y.tolist (), labels.tolist ()))
  
  if doPlot:
    figure =  plt.figure (1, figsize=(4, 3))
    
    ax = Axes3D(figure, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X.iloc[:, 2], X.iloc[:, 0], X.iloc[:, 1], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Brand Price Ratio')
    ax.set_ylabel('Brand vs Non-Brand Ratio')
    ax.set_zlabel('Unit Price')

    plt.show ()
    plt.savefig (utilities.outputFolder + description + '-' + str (numClusters) + '-cluster')
    print 'Clusters saved to output folder.'
    plt.close (figure)

  return productCodesToClusters


def GetHouseholdProductFrequencyByCode (productCode, saveFiles=False):
  """This method returns a table specifying how many brand vs non-brand
     purchases each household had against a specifict product code.

  The final result should look like this:
  household_code | brand purchases | non-brand purchases | ratio

  Keyword Arguments:
  productCode - unique product code from dataset
  saveFiles - if True we will pickleize the final result and plot a histogram
             of the ratios
  """
  # Search for all products with the specified product code
  productSearch = utilities.products[utilities.products['product_module_code'] == productCode]

  # Merge the products dataset with the purchases dataset
  merged = pd.merge (productSearch, utilities.purchases, on=None, left_on=None, right_on='upc', left_index=True, right_index=False)

  # At this point, our index is the unique trip code id, we want 
  # to preserve these values as a column
  merged['trip_code_uc'] = merged.index.tolist ()

  # Label all products with CTL in their description as non-brand
  merged['is_brand'] = np.where (merged.brand_descr.str.contains ('CTL') == True, False, True)

  # Now we merge our current dataset with the trips dataset.
  # The purpose is to identify all households that bought a specific product
  merged = pd.merge (merged, utilities.trips, on=None, left_on='trip_code_uc', right_on=None, left_index=False, right_index=True)

  # Group all households and the brand column with the purpose of 
  # getting a sum of all brand and non-brand purchases for each household
  merged = merged.groupby (['household_code', 'is_brand'])['quantity'].sum ().reset_index ()

  # Seperate each brand and non-brand purchase into its own column
  merged['brand_purchases'] = np.where (merged.is_brand == True, merged.quantity, 0)
  merged['non_brand_purchases'] = np.where (merged.is_brand == False, merged.quantity, 0)

  # Add up all the brand and non-brand purchases together
  merged = merged.groupby (['household_code'])[['brand_purchases', 'non_brand_purchases']].sum ().reset_index ()

  # Get the ratio between brand and non-brand purchases
  merged['ratio'] = merged.brand_purchases.astype(float) / (merged.brand_purchases.astype(float) + merged.non_brand_purchases.astype(float))

  if saveFiles:
    # We only save files if the household bought any products.
    if len (merged.ratio) > 0:
      merged.to_pickle (utilities.brandFrequencyFolder + str (productCode))

      fig = plt.figure ()
      productName = productHierarchy[productHierarchy['product_module_code'] == productCode].product_module_descr.tolist ()[0]
      plt.title (str (productName) + ' - Ratio of Brand Purchases')
      plt.xlabel ('Ratio')
      plt.ylabel ('Frequency')
      merged.ratio.plot.hist ()
      plt.savefig (utilities.brandFrequencyFolder + str (productCode) + '.png')
      plt.close (fig)

  return merged

def PrepareDataInternal (productCode):
  """This method merges the data with the panelist dataset. The intent
     is to get more information of each household. We then drop everything 
     we don't need and transform various fields.

  Keyword Arguments:
  productCode - integer of product code to prepare data for
  """
  householdProductFrequency = GetHouseholdProductFrequencyByCode (productCode)

  # Merge with the panelist file and drop all the data we don't want to use
  merged = pd.merge (householdProductFrequency, utilities.panelist, on=None, left_on='household_code', right_on=None, left_index=False, right_index=True)
  merged = merged.drop \
        ([ \
          'panel_year', 'projection_factor', 'projection_factor_magnet', 'household_composition', \
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

  # Convert the following categorical varaibles to booleans
  merged[merged['age_and_presence_of_children'] == 9] = 0
  merged[merged['male_head_employment'] == 9] = 0
  merged[merged['female_head_employment'] == 9] = 0
  merged['hispanic_origin'] = np.where (merged.hispanic_origin == 1, 1, 0)
  merged['wic_indicator_current'] = np.where (merged.wic_indicator_current == 1, 1, 0)

  # Seperate out race into boolean variables
  merged['race_white'] = np.where (merged.race == 1, 1, 0)
  merged['race_black'] = np.where (merged.race == 2, 1, 0)
  merged['race_asian'] = np.where (merged.race == 3, 1, 0)
  merged['race_other'] = np.where (merged.race == 4, 1, 0)
  merged = merged.drop ('race', 1)
  
  # Seperate out our labels from the features
  labels = pd.DataFrame (merged['ratio'])
  merged = merged.drop ('ratio', 1)

  return (merged, labels)

def PrepareData (productCode, cutOff, scale=True, numClusters=2):
  """This method prepares the data before running it through any algorithm.
     We optionally scale the data.

  Keyword Arguments:
  productCode - Can be a string: 'a,b,c' or an int: a
  cutOff - This will be the cutoff for branded vs non-branded buyers
  scale - If true we scale the data using a standard scaler
  """
  if ',' in productCode:
    productCode = productCode.split (',')
    X = Y = pd.DataFrame ()
  else:
    (X, Y) = PrepareDataInternal (int (productCode))

  if type (productCode) is list:
    productDescriptions = []
    for pc in productCode:
      productDescriptions.append \
          (utilities.productHierarchy[utilities.productHierarchy['product_module_code'] == int (pc)].product_group_descr.to_string (index=False))

    productDescriptions = list (set (productDescriptions))

  if CLUSTER_FEATURE:
    productsToClusters = ClusterInternal ("::".join (productDescriptions), True, numClusters, False)
  elif PRODUCT_FEATURE:
    XProd = YProd = pd.DataFrame ()
    for descr in productDescriptions:
      (XProdNew, YProdNew) = GetProductFeatures (descr, True)
      YProdNew = pd.DataFrame (YProdNew)
      XProd = pd.concat ([XProd, XProdNew])
      YProd = pd.concat ([YProd, YProdNew])
    XProd = XProd.drop ('brand_ratio', 1)

    productFeatures = pd.concat ([XProd, YProd], 1)
    del XProd, YProd

  if type (productCode) is list:
    for pc in productCode:
      (xNew, yNew) = PrepareDataInternal (int (pc))

      if CLUSTER_FEATURE:
        currentCluster = productsToClusters[int (pc)]
        clustersColumn = []
        for j in range (0, xNew.shape[0]):
          clustersColumn.append(currentCluster)
        xNew['cluster'] = pd.Series (clustersColumn).values
      elif PRODUCT_FEATURE:
        xNew['unit_price'] = pd.Series (
            np.ones (xNew.shape[0]) * \
            productFeatures.loc[productFeatures.product_module_code == int (pc)]['unit_price'].iloc[0] \
            )

        xNew['brand_price_ratio'] = pd.Series (
            np.ones (xNew.shape[0]) * \
            productFeatures.loc[productFeatures.product_module_code == int (pc)]['brand_price_ratio'].iloc[0] \
            )

      X = pd.concat ([X, xNew])
      Y = pd.concat ([Y, yNew])

  # Preserve column names before applying scaling
  columnNames = X.columns.values.tolist ()

  print columnNames

  if scale == True:
    scaler = StandardScaler ()
    # This operation will not keep the column names
    X = pd.DataFrame (scaler.fit_transform (X))

  X.columns = columnNames

  if MULTINOMIAL_PREDICTION:
    Y[Y['ratio'] <= 0.33] = 0
    Y[(Y['ratio'] > 0.33) & (Y['ratio'] <= 0.66)] = 1
    Y[Y['ratio'] > 0.66] = 2
  else:
    Y['ratio'] = np.where (Y.ratio >= cutOff, 1, 0)

  return (X, Y)

def PrepareDataNoCluster (productCode, cutOff):
  if ',' in productCode:
    productCode = productCode.split (',')
    X = Y = pd.DataFrame ()
  else:
    (X, Y) = PrepareDataInternal (int (productCode))

  if type (productCode) is list:
    for pc in productCode:
      (xNew, yNew) = PrepareDataInternal (int (pc))
      xNew['product_code'] = np.ones (xNew.shape[0]) * int (pc)
      X = pd.concat ([X, xNew])
      Y = pd.concat ([Y, yNew])

  # Preserve column names before applying scaling
  columnNames = X.columns.values.tolist ()

  X.columns = columnNames

  if MULTINOMIAL_PREDICTION:
    Y[Y['ratio'] <= 0.33] = 0
    Y[(Y['ratio'] > 0.33) & (Y['ratio'] <= 0.66)] = 1
    Y[Y['ratio'] > 0.66] = 2
  else:
    Y['ratio'] = np.where (Y.ratio >= cutOff, 1, 0)

  return (X, Y)
