from osgeo import gdal
import numpy as np
ds = gdal.Open('data/93f2f30620ff4350b38573a4f4b68351')
mask = ds.GetRasterBand(1).ReadAsArray()
from matplotlib import pyplot as plt
plt.imshow(mask)
plt.show()


#mask = mask.astype('float')
#mask[mask == 0] = 'nan'
#mask[mask == 1] = 'nan'

#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(strategy="most_frequent")
#from sklearn.impute import KNNImputer
#imp = KNNImputer()

#a = imp.fit_transform(mask)

from skimage import filters

from skimage.morphology import disk, ball
#mask_null = np.isnan(mask)
#filled_image = np.where(mask_null, a, mask)
filled_image = filters.rank.modal(mask, disk(10))



plt.imshow(filled_image)
plt.show()
