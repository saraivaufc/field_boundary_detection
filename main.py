from sam_service import SAMService
from settings import  SEGMENTATION_SAM_MODEL_CONFIG

field_service = SAMService(SEGMENTATION_SAM_MODEL_CONFIG)

#field_service.predict_chips('data/input2.tif', 'data/output2.gpkg') # real	1m17,611s

#field_service.predict_chips('data/input.tif', 'data/output.gpkg') # real	105m54,310s

field_service.predict_chips('data/CBERS_4A_WPM_20231015_204_130_L4_BAND0.tif', 'data/CBERS_4A_WPM_20231015_204_130_L4_BAND0.gpkg')
