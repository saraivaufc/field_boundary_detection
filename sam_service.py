import os
import gc
import numpy as np
import torch
from osgeo import gdal, osr, ogr
import tempfile
import settings

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

gdal.UseExceptions()

largura_pedaco = 1000
altura_pedaco = 1000
input_Images = 'data'


class SAMService:

    def __init__(self, config):
        sam_checkpoint = config.get('path')
        model_type = config.get('type')
        device = config.get('device')

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.__mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=1)

    def normalize_image(self, image):
        image_min = np.min(image)
        image_max = np.max(image)
        image = np.array(((image - image_min) / (image_max - image_min)) * 255).astype(np.uint8)
        return image

    def get_masks(self, image):
        torch.cuda.empty_cache()
        return self.__mask_generator.generate(image)

    def convert_masks_to_full_mask(self, image, anns):
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        full_image = np.zeros((image.shape[0], image.shape[1], 1))

        id = 1
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 1))
            img[:, :] = id
            stacked_image = img * m.reshape((m.shape[0], m.shape[1], 1))
            full_image = np.maximum(full_image, stacked_image)
            del stacked_image
            gc.collect()
            id += 1
        return full_image

    def save_full_mask(self, image_dataset, mask, output_path):
        mem_dataset = image_dataset \
            .GetDriver() \
            .Create(output_path, mask.shape[1], mask.shape[0], 1, gdal.GDT_Int16)

        mem_dataset.SetGeoTransform(image_dataset.GetGeoTransform())
        mem_dataset.SetProjection(image_dataset.GetProjection())
        mem_dataset.GetRasterBand(1) \
            .WriteArray(mask.reshape((mask.shape[0], mask.shape[1])), 0, 0)
        mem_dataset.FlushCache()
        return output_path

    def convert_raster_to_vector(self, input_raster_path, output_vector_path):
        # load raster band
        raster = gdal.Open(input_raster_path)
        raster_band = raster.GetRasterBand(1)

        # get raster projection
        projection = osr.SpatialReference()
        projection.ImportFromWkt(raster.GetProjectionRef())

        # get Shapefile driver
        shp_driver = ogr.GetDriverByName("GPKG")

        # create shapefile dataset
        if os.path.exists(output_vector_path):
            shp_driver.DeleteDataSource(output_vector_path)

        # create layer
        new_shp = shp_driver.CreateDataSource(output_vector_path)
        new_shp.CreateLayer('mask', geom_type=ogr.wkbPolygon, srs=projection)

        # get layer
        dst_layer = new_shp.GetLayer()

        # create new field to define values
        new_field = ogr.FieldDefn('id', ogr.OFTInteger)
        dst_layer.CreateField(new_field)

        # Polygonize(band, hMaskBand[optional]=None, destination lyr, field ID, papszOptions=[], callback=None)
        gdal.Polygonize(raster_band, None, dst_layer, 0, [], callback=None)

    def predict(self, input_image_path: str, output_fields_path: str):
        print(f'Loading image {input_image_path}  ...')
        image_dataset = gdal.Open(input_image_path)

        print(image_dataset.GetRasterBand(1).GetStatistics(True, True))

        print(image_dataset.GetMetadata())

        image_array = image_dataset.ReadAsArray().transpose(1, 2, 0)
        gc.collect()

        print('Normalizing image...')
        image_array = self.normalize_image(image_array)
        gc.collect()

        print('Generating masks...')
        masks = self.get_masks(image_array)
        gc.collect()

        print('Generating full mask...')
        full_mask = self.convert_masks_to_full_mask(image_array, masks)
        gc.collect()

        print('Exporting full mask...')
        image_filename, image_extension = os.path.splitext(input_image_path)
        mask_output_path = f'{image_filename}_mask{image_extension}'
        output_mask_path = self.save_full_mask(image_dataset, full_mask, mask_output_path)
        gc.collect()

        print('Vetorizing mask...')
        self.convert_raster_to_vector(output_mask_path, output_fields_path)

    def predict_chips(self, input_file, output_file, reference=osr.SpatialReference()):
        image_dataset = gdal.Open(input_file)

        geotransform = image_dataset.GetGeoTransform()
        x_origin = geotransform[0]
        y_origin = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

        largura_imagem = image_dataset.RasterXSize
        altura_imagem = image_dataset.RasterYSize

        num_pedacos_x = int(largura_imagem / largura_pedaco) + (largura_imagem % largura_pedaco > 0)
        num_pedacos_y = int(altura_imagem / altura_pedaco) + (altura_imagem % altura_pedaco > 0)

        with tempfile.TemporaryDirectory() as tmpdirname:

            fields_driver = ogr.GetDriverByName('GPKG')
            fields_dataset = fields_driver.CreateDataSource(output_file)

            proj = osr.SpatialReference(wkt=image_dataset.GetProjection())

            fields_layer = fields_dataset.CreateLayer('main',
                                                      geom_type=ogr.wkbPolygon,
                                                      srs=proj)
            print(f'fields_layer: {fields_layer}')
            for i in range(num_pedacos_x):
                for j in range(num_pedacos_y):
                    x_offset = i * largura_pedaco
                    y_offset = j * altura_pedaco
                    x_size = min(largura_pedaco, largura_imagem - x_offset)
                    y_size = min(altura_pedaco, altura_imagem - y_offset)

                    x_upper_left = x_origin + x_offset * pixel_width
                    y_upper_left = y_origin + y_offset * pixel_height
                    x_lower_right = x_upper_left + x_size * pixel_width
                    y_lower_right = y_upper_left + y_size * pixel_height

                    pedaco_geotransform = (x_upper_left, pixel_width, 0, y_upper_left, 0, pixel_height)

                    pedaco = image_dataset.ReadAsArray(x_offset, y_offset, x_size, y_size)

                    filename, _ = os.path.splitext(os.path.basename(input_file))

                    output_chip_path = f'{tmpdirname}/{filename}_{i}_{j}.tif'

                    driver = gdal.GetDriverByName("GTiff")
                    pedaco_ds = driver.Create(output_chip_path, x_size, y_size, image_dataset.RasterCount,
                                              image_dataset.GetRasterBand(1).DataType)

                    pedaco_ds.SetGeoTransform(pedaco_geotransform)
                    pedaco_ds.SetProjection(image_dataset.GetProjection())

                    for k in range(image_dataset.RasterCount):
                        band = pedaco_ds.GetRasterBand(k + 1)
                        band.WriteArray(pedaco[k, :, :])
                        band.FlushCache()
                        del band

                    pedaco_ds.FlushCache()
                    del pedaco_ds

                    output_fields_path = output_chip_path.replace('.tif', '.gpkg')
                    self.predict(output_chip_path, output_fields_path)

                    chip_dataset = ogr.Open(output_fields_path)
                    chip_layer = chip_dataset.GetLayer()
                    for chip_feature in chip_layer:
                        out_feat = ogr.Feature(fields_layer.GetLayerDefn())
                        out_feat.SetGeometry(chip_feature.GetGeometryRef().Clone())
                        fields_layer.CreateFeature(out_feat)
                        fields_layer.SyncToDisk()