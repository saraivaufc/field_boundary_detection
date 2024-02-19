from sam_service import SAMService
from settings import  SEGMENTATION_SAM_MODEL_CONFIG

field_service = SAMService(SEGMENTATION_SAM_MODEL_CONFIG)

# real	1m17,611s
#field_service.predict_chips('data/input2.tif', 'data/output2.gpkg')

# real	105m54,310s
#field_service.predict_chips('data/input.tif', 'data/output.gpkg')

#real	377m2,864s
#field_service.predict_chips('data/CBERS_4A_WPM_20230709_205_128_L4_BAND0.tif', 'data/CBERS_4A_WPM_20230709_205_128_L4_BAND0.gpkg')

field_service.predict_chips('data/CBERS_4A_WPM_20230709_205_127_L4_BAND0.tif',
                            'data/CBERS_4A_WPM_20230709_205_127_L4_BAND0.gpkg')


field_service.predict_chips('data/CBERS_4A_WPM_20230804_206_127_L4_BAND0.tif',
                            'data/CBERS_4A_WPM_20230804_206_127_L4_BAND0.gpkg')

field_service.predict_chips('data/CBERS_4A_WPM_20230804_206_128_L4_BAND0.tif',
                            'data/CBERS_4A_WPM_20230804_206_128_L4_BAND0.gpkg')