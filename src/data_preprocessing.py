
import utilities

import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
  # merged['age_and_presence_of_children'] = np.where (merged.age_and_presence_of_children == 9, 0, 1)
  merged[merged['age_and_presence_of_children'] == 9] = 0
  # merged['male_head_employment'] = np.where (merged.male_head_employment.isin ([1, 2, 3]), 1, 0)
  # merged['female_head_employment'] = np.where (merged.female_head_employment.isin ([1, 2, 3]), 1, 0)
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

def PrepareData (productCode, cutOff, scale=True):
  """This method prepares the data before running it through any algorithm.
     We optionally scale the data.

  Keyword Arguments:
  productCode - Can be a string: 'a,b,c' or an int: a
  cutOff - This will be the cutoff for branded vs non-branded buyers
  scale - If true we scale the data using a standard scaler
  """
  if ',' in productCode:
    productCode = productCode.split (',')
    (X, Y) = pd.DataFrame ()
  else:
    (X, Y) = PrepareDataInternal (int (productCode))

  # This is for the case where productCode is a string, here
  # we concatenate data from multiple products toguether 
  # TODO RARS - This might not be the right thing to do, think more
  for i in range (0, len (productCode)):
    (xNew, yNew) = PrepareDataInternal (int (productCode[i]))
    X = pd.concat ([X, xNew])
    Y = pd.concat ([Y, yNew])

  if scale == True:
    scaler = StandardScaler ()
    X = pd.DataFrame (scaler.fit_transform (X))

  Y['ratio'] = np.where (Y.ratio > cutOff, 1, 0)

  return (X, Y)
