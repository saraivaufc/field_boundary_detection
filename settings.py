import os

SEGMENTATION_SAM_MODEL_CONFIG = {
    'path': os.environ.get('SEGMENTATION_SAM_MODEL_PATH', 'sam_vit_l_0b3195.pth'),
    'type': os.environ.get('SEGMENTATION_SAM_MODEL_TYPE', 'vit_l'),
    'device': os.environ.get('SEGMENTATION_SAM_MODEL_DEVICE', 'cuda'),
}
