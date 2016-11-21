
import time
import pandas as pd
import cPickle as pickle

global dataFolder, outputFolder, pickleFolder, brandFrequencyFolder
dataFolder = 'data/'
outputFolder = 'output/'
pickleFolder = 'pickled-objects/'
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
def DatasetToPickle ():
  """Converts our dataset to pickle format. This makes it easier
     to read in the future
  """
  brandVariations = pd.read_csv (dataFolder + 'brand_variations.tsv', sep='\t', index_col=0)
  products = pd.read_csv (dataFolder + 'products.tsv', sep='\t', index_col=0)
  purchases = pd.read_csv (dataFolder + 'purchases_2014.tsv', sep='\t', index_col=0)
  trips = pd.read_csv (dataFolder + 'trips_2014.tsv', sep='\t', index_col=0)
  panelist = pd.read_csv (dataFolder + 'panelists_2014.tsv', sep='\t', index_col=0)
  productsExtra = pd.read_csv (dataFolder + 'products_extra_2014.tsv', sep='\t', index_col=0, low_memory=False)
  retailers = pd.read_csv (dataFolder + 'retailers.tsv', sep='\t', index_col=0)
  productHierarchy = pd.read_excel (dataFolder + 'Product_Hierarchy_1.22.2016.xlsx')
  dmaToFips = pd.read_excel (dataFolder + '2014 DMA to FIPS conversion table.xlsx')

  brandVariations.to_pickle (pickleFolder + 'brand-variations.pkl')
  products.to_pickle (pickleFolder + 'products.pkl')
  purchases.to_pickle (pickleFolder + 'purchases.pkl')
  trips.to_pickle (pickleFolder + 'trips.pkl')
  panelist.to_pickle (pickleFolder + 'panelist.pkl')
  productsExtra.to_pickle (pickleFolder + 'products-extra.pkl')
  retailers.to_pickle (pickleFolder + 'retailers.pkl')
  productHierarchy.to_pickle (pickleFolder + 'product-hierarchy.pkl')
  dmaToFips.to_pickle (pickleFolder + 'dma-to-fips.pkl')

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
