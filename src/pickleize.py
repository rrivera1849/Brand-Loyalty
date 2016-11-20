
import pandas as pd
import cPickle as pickle

dataFolder = 'data/'
pickleFolder = 'pickled-objects/'

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

def main ():
  DatasetToPickle ()

if __name__ == "__main__":
  main ()
