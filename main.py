from sam_service import SAMService
from settings import  SEGMENTATION_SAM_MODEL_CONFIG

field_service = SAMService(SEGMENTATION_SAM_MODEL_CONFIG)

field_service.predict_chips('data/CBERS_4A_WPM_20230809_205_127_L4_RGB_CLIPPED.tif', 'data/CBERS_4A_WPM_20230809_205_127_L4_RGB_CLIPPED.gpkg')
