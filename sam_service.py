import os
import gc
import numpy as np
import torch
import uuid
from osgeo import gdal, osr, ogr
from skimage import filters
from skimage.morphology import disk
import geopandas as gpd
from shapely.validation import make_valid

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

    def remove_bordes(self, image):
        image = image.reshape((image.shape[0], image.shape[1])).astype(int)
        filled_image = filters.rank.modal(image, disk(10))
        return filled_image.reshape((filled_image.shape[0], filled_image.shape[1], 1))


    def save_full_mask(self, image_dataset, mask):
        # MEM
        # GTiff
        mem_dataset = gdal.GetDriverByName("MEM") \
            .Create(uuid.uuid4().hex, mask.shape[1], mask.shape[0], 1, gdal.GDT_Int16)

        mem_dataset.SetGeoTransform(image_dataset.GetGeoTransform())
        mem_dataset.SetProjection(image_dataset.GetProjection())
        mem_dataset.GetRasterBand(1) \
            .WriteArray(mask.reshape((mask.shape[0], mask.shape[1])), 0, 0)
        mem_dataset.FlushCache()
        return mem_dataset

    def convert_raster_to_vector(self, full_mask, output_vector_path):
        # reshape mask
        raster_band = full_mask.GetRasterBand(1)

        # get raster projection
        projection = osr.SpatialReference()
        projection.ImportFromWkt(full_mask.GetProjectionRef())

        # create layer
        new_shp = ogr.GetDriverByName("Memory").CreateDataSource(output_vector_path)
        new_shp.CreateLayer('mask', geom_type=ogr.wkbPolygon, srs=projection)

        # get layer
        dst_layer = new_shp.GetLayer()

        # create new field to define values
        new_field = ogr.FieldDefn('id', ogr.OFTInteger)
        dst_layer.CreateField(new_field)

        gdal.Polygonize(raster_band, None, dst_layer, 0, [], callback=None)
        return new_shp

    def remove_stripes(self, old_final_gdf):
        final_gdf = old_final_gdf.reset_index(drop=True).explode(
            ignorar_index=True,
            index_parts=True)

        final_gdf['geometry_type'] = final_gdf.geometry.geom_type
        final_gdf = final_gdf[final_gdf['geometry_type'].isin(['Polygon', 'MultiPolygon'])]

        final_gdf['area_ha'] = final_gdf.area / 10000
        final_gdf = final_gdf.sort_values('area_ha')

        final_gdf['id'] = range(len(final_gdf))
        final_gdf['new_id'] = final_gdf['id']

        gdf_overlaps = gpd.overlay(final_gdf, final_gdf, how='intersection', keep_geom_type=True, make_valid=True)
        gdf_overlaps['intersection_area_ha'] = gdf_overlaps['geometry'].area / 10000
        gdf_overlaps = gdf_overlaps[
            (gdf_overlaps['id_1'] != gdf_overlaps['id_2']) & (gdf_overlaps['intersection_area_ha'] > 0.01)]

        gdf_overlaps['id_min'] = gdf_overlaps.apply(lambda x: min(x['id_1'], x['id_2']), axis=1)
        gdf_overlaps['id_max'] = gdf_overlaps.apply(lambda x: max(x['id_1'], x['id_2']), axis=1)

        gdf_overlaps = gdf_overlaps.drop_duplicates(['id_min', 'id_max', 'intersection_area_ha'])

        while True:
            gdf_overlaps = gdf_overlaps[(gdf_overlaps['id_min'] != gdf_overlaps['id_max'])]

            if gdf_overlaps.empty:
                break

            for id in list(gdf_overlaps['id_min'].unique()):
                intersections_rows = gdf_overlaps[(gdf_overlaps['id_min'] == id)]

                id_max = intersections_rows.sort_values('intersection_area_ha', ascending=False)['id_max'].to_list()[0]

                gdf_overlaps.loc[(gdf_overlaps['id_min'] == id) & (gdf_overlaps['id_max'] == id_max), 'id_max'] = id

                final_gdf.loc[final_gdf['id'] == id, 'new_id'] = id_max

            final_gdf['geometry'] = final_gdf['geometry'].apply(make_valid)
            final_gdf = final_gdf.dissolve(by=['new_id'], aggfunc='max', as_index=False)
            final_gdf['area_ha'] = final_gdf.area / 10000

        return final_gdf

    def predict(self, image_dataset, output_fields_path: str):
        print('Get array')
        image_array = image_dataset.ReadAsArray().transpose(1, 2, 0)

        print('Normalizing image...')
        image_array = self.normalize_image(image_array)

        print('Generating masks...')
        masks = self.get_masks(image_array)

        print('Generating full mask...')
        full_mask = self.convert_masks_to_full_mask(image_array, masks)

        print('Remove border full mask...')
        full_mask = self.remove_bordes(full_mask)

        print('Exporting full mask...')
        full_mask_image = self.save_full_mask(image_dataset, full_mask)

        return self.convert_raster_to_vector(full_mask_image, output_fields_path)

    def predict_chips(self, input_file, output_file):
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

        output_file_prefix, output_file_extension = output_file.split('.')
        output_file_raw = f'{output_file_prefix}_raw.{output_file_extension}'

        fields_driver = ogr.GetDriverByName('GPKG')
        fields_dataset = fields_driver.CreateDataSource(output_file_raw)

        proj = osr.SpatialReference(wkt=image_dataset.GetProjection())

        fields_layer = fields_dataset.CreateLayer('main',
                                                  geom_type=ogr.wkbPolygon,
                                                  srs=proj)
        print(f'fields_layer: {fields_layer}')
        for i in range(num_pedacos_x):
            for j in range(num_pedacos_y):
                margin = 1
                x_offset = (i * largura_pedaco)
                y_offset = (j * altura_pedaco)
                x_size = min(largura_pedaco, largura_imagem - x_offset)
                y_size = min(altura_pedaco, altura_imagem - y_offset)

                x_size_margin = x_size + margin
                y_size_margin = y_size + margin

                if (x_offset + x_size_margin) > largura_imagem:
                    x_size_final = x_size
                else:
                    x_size_final = x_size_margin

                if (y_offset + y_size_margin) > altura_imagem:
                    y_size_final = y_size
                else:
                    y_size_final = y_size_margin

                print('X', x_size, x_size_margin, (x_offset + x_size_margin), largura_imagem)
                print('Y', y_size, y_size_margin, (y_offset + y_size_margin), altura_imagem)

                x_upper_left = x_origin + x_offset * pixel_width
                y_upper_left = y_origin + y_offset * pixel_height

                pedaco_geotransform = (x_upper_left, pixel_width, 0, y_upper_left, 0, pixel_height)

                pedaco = image_dataset.ReadAsArray(x_offset, y_offset, x_size_final, y_size_final)

                filename, _ = os.path.splitext(os.path.basename(input_file))

                output_chip_path = f'{filename}_{i}_{j}.tif'

                driver = gdal.GetDriverByName("MEM")
                pedaco_ds = driver.Create(output_chip_path, x_size_final, y_size_final, image_dataset.RasterCount,
                                          image_dataset.GetRasterBand(1).DataType)

                pedaco_ds.SetGeoTransform(pedaco_geotransform)
                pedaco_ds.SetProjection(image_dataset.GetProjection())

                for k in range(image_dataset.RasterCount):
                    band = pedaco_ds.GetRasterBand(k + 1)
                    band.WriteArray(pedaco[k, :, :])
                    band.FlushCache()
                    del band

                pedaco_ds.FlushCache()

                output_fields_path = output_chip_path.replace('.tif', '.gpkg')

                chip_dataset = self.predict(pedaco_ds, output_fields_path)

                chip_layer = chip_dataset.GetLayer()
                for chip_feature in chip_layer:
                    out_feat = ogr.Feature(fields_layer.GetLayerDefn())
                    out_feat.SetGeometry(chip_feature.GetGeometryRef().Clone())
                    fields_layer.CreateFeature(out_feat)
                    fields_layer.SyncToDisk()

        raw_gdf = gpd.read_file(output_file_raw)
        final_gdf = self.remove_stripes(raw_gdf)
        final_gdf.to_file(output_file)