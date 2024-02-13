from sam_service import SAMService
from settings import  SEGMENTATION_SAM_MODEL_CONFIG

field_service = SAMService(SEGMENTATION_SAM_MODEL_CONFIG)
field_service.predict_chips('data/input.tif', 'data/fields.gpkg')
