################################
import numpy as np

import pandas as pd

# all imports should go here

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show

import sklearn

import skimage.exposure

# access package for AWS access
# import boto3

import sys
import os
import subprocess
import datetime
import platform
import datetime

from tqdm import tqdm

# import ee
import h5py
import numpy as np
from datetime import datetime, timedelta  # Import timedelta here
import random
import pandas as pd

import time
import rasterio as rio
################################

import shutil

# Execute only once!
import os
import sys

class HLSDataSet:
    def __init__(self, table_dtype = 'float16', path='./aispace/data/L8-100x100'):
        self.data_path = path
        self.input_data = pd.read_csv(self.data_path)

        self.input_data.replace(-9999, np.nan, inplace=True)
        self.input_data['NDVI'] = (self.input_data['B05'] - self.input_data['B04']) / \
                                  (self.input_data['B05'] + self.input_data['B04']) 



        # Add a new column 'PID' or 'Point_ID' with unique IDs for each point
        self.input_data['PID'] = self.input_data.groupby(['X', 'Y']).ngroup()

        print('PIDs is ', max(sorted(self.input_data['PID'].unique())))

        self.input_data = self.input_data.reset_index(drop=True)
        
        self.BND_LIST = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B09', 'B10', 'B11']
        self.FMSK_LIST = ['cirrus', 'cloud', 'adj_cloud', 'cloud_shadow', 'snow_ice', 'water', 'aero' ]
        self.ANG_LIST = ['SAA', 'SZA', 'VAA', 'VZA',]

        self.REFLECT = False

        self.data = None
        self.nan_data = None
        self.clear_data = None
        self.cloud_data = None

        #### train + test = clear always!!!! ###################
        self.test_data = None
        self.train_data = None

        self.to_impute = None
        self.imputed_data = None
        self.inference_data = None

        self.table_dtype = table_dtype
        
    def clip_dataset(self, x1, y1, x2, y2):

        x_shift = self.input_data['X'].min()
        y_shift = self.input_data['Y'].min()
        
        self.input_data = self.input_data.loc[(self.input_data['X'] >= x1 + x_shift) & \
                                              (self.input_data['X'] <= x2 + x_shift) & \
                                              (self.input_data['Y'] >= y1 + y_shift) & \
                                              (self.input_data['Y'] <= y2 + y_shift)].copy()
                                              
        self.input_data = self.input_data.reset_index(drop=True)
        
        return self.input_data


    def _REFLECTANCE(self, bnd_list=['B04', 'B03', 'B02'], round = 4, data_type='float'):

        self.REFLECT = True

        min_val = -0.063
        max_val = 0.3

        for i_band in bnd_list:
            self.input_data[i_band] = 0.0001 * self.input_data[i_band]
            # Set all elements in column 'B0' greater than 0.3 to 0.3
            self.input_data[i_band] = self.input_data[i_band].apply(lambda x: min(x, max_val))
            self.input_data[i_band] = (self.input_data[i_band] - min_val) / (max_val - min_val) 
            self.input_data[i_band] = self.input_data[i_band].round(round)
            if data_type == 'int':
                self.input_data[i_band] = self.input_data[i_band] * 10**round
                self.input_data[i_band] = self.input_data[i_band].astype(int)

    def _QUANTIZATE(self, bnd_list=['B04', 'B03', 'B02'], round = 4, data_type='int'):

        self.REFLECT = True

        min_val = -0.063
        max_val = 0.3

        for i_band in bnd_list:
            self.input_data[i_band] = 0.0001 * self.input_data[i_band]
            # Set all elements in column 'B0' greater than 0.3 to 0.3
            self.input_data[i_band] = np.where(self.input_data[i_band] > max_val, max_val, self.input_data[i_band])
            # self.input_data[i_band] = self.input_data[i_band].apply(lambda x: min(x, max_val))
            self.input_data[i_band] = (self.input_data[i_band] - min_val) / (max_val - min_val) 
            self.input_data[i_band] = self.input_data[i_band].round(round)
            if data_type == 'int':
                # self.input_data[i_band] = self.input_data[i_band] * 10**round
                self.input_data[i_band] = ( self.input_data[i_band] * 255 )  #.astype(np.uint8)


    def _image_df(self, input):

        input = input
        
        box_x_size = (input['X'].max() - input['X'].min() + 1).astype('int')
        box_y_size = (input['Y'].max() - input['Y'].min() + 1).astype('int')
    
        def _get_img_nan(input, bnd_list=['B04', 'B03', 'B02']):
    
            print(input['DOY'].unique())
    
            df = input[bnd_list].copy()
    
            df[df > 0] = 0
            df[df == -9999] = 1
            df[df < 0] = 0
    
            image = df.to_numpy()  # df[chanel_list]
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            nans = np.dstack((image[0,:,:], image[1,:,:], image[2,:,:]))
    
            return nans
    
    
        def _get_img_rgb(input, bnd_list=['B04', 'B03', 'B02']):
            df = input[bnd_list].copy()
    
            # df[df > 0] = 0
            df[df == -9999] = np.nan
            # df[df < 0] = 0
    
            image = df.to_numpy()
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            # Convert the int16 array to int64
            # image = image.astype(np.uint64)
    
            def generalized_normalization(band):
                # Apply your normalization method here
                # Example: Stretch and scale values to 0-255
                band = np.ma.array (band, mask=np.isnan(band))

                if self.REFLECT == False:
                    ### FOR HLS #################
                    band = 0.0001 * band
                    band = np.where(band > 0.3, 0.3, band)
                    min_val = -0.063
                    max_val = 0.3
                    
                    normalized_band = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:                
                    normalized_band = band

                    normalized_band[normalized_band == np.nan] = 0 #255
                    normalized_band = band.astype(np.uint8)
                return normalized_band
    
            # Scale the bands to 8-bit
            scaled_red = generalized_normalization(image[2,:,:])
            scaled_green = generalized_normalization(image[1,:,:])
            scaled_blue = generalized_normalization(image[0,:,:])
     
            rgb = np.dstack((scaled_red, scaled_green, scaled_blue))
    
            return rgb

        def _get_img_ndvi(input, bnd_list=['NDVI']):
            df = input[bnd_list].copy()
    
            # df[df > 0] = 0
            df[df == -9999] = np.nan
            # df[df < 0] = 0
    
            image = df.to_numpy()
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            # Convert the int16 array to int64
            # image = image.astype(np.uint64)
    
            r_ = (8200, 16000)
            g_ = (8500, 14000)
            b_ = (7500, 12000)
    
    
            def ndvi_normalization(band):
                # Apply your normalization method here
                # Example: Stretch and scale values to 0-255
                band = np.ma.array (band, mask=np.isnan(band))
    
                ### FOR HLS #################
                min_val = 0.0
                max_val = 1.0
    
                normalized_band = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                normalized_band[normalized_band == np.nan] = 255
                return normalized_band
    
            # Scale the bands to 8-bit
            scaled_ndvi = ndvi_normalization(image[0,:,:])
    
            return scaled_ndvi
    
        def _get_img(input, bnd_list):
            image = input[bnd_list].to_numpy()  # df[chanel_list]
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            # Convert the int16 array to int64
            image = image.astype(np.uint64)
    
            return image[0,:,:]
    
        image_nan = _get_img_nan(input, bnd_list=['B02', 'B03', 'B04'])
        image_rgb = _get_img_rgb(input, bnd_list=['B02', 'B03', 'B04'])
        image_cirrus = _get_img(input, bnd_list=['cirrus'])
        image_cloud = _get_img(input, bnd_list=['cloud'])
        image_adjcloud = _get_img(input, bnd_list=['adj_cloud'])
        image_cloud_shadow = _get_img(input, bnd_list=['cloud_shadow'])
        image_snow_ice = _get_img(input, bnd_list=['snow_ice'])
        image_water = _get_img(input, bnd_list=['water'])

        image_ndvi = _get_img_ndvi(input, bnd_list=['NDVI'])
    
        image_rgb_list = [image_rgb, image_ndvi, image_cloud, image_adjcloud, image_cloud_shadow, image_snow_ice, image_water]
    
        # Create subplots
        fig, axes = plt.subplots(1, len(image_rgb_list), figsize=(18, 22))
        # Flatten the axes array to simplify indexing
        axes = axes.ravel()
        # print(image_rgb_list[0].shape)
        axes[0].imshow(image_rgb_list[0])
        # Loop through the images and plot them
        for ii in range(1,len(image_rgb_list)):
            axes[ii].imshow(image_rgb_list[ii], cmap='gray')  # You can specify a colormap
        plt.tight_layout()
        plt.show()
    

    def _get_data_doys(self, doys = [171, 179, 187, 195, 203, 211, 219], SHOW=True):
        train_data = self.input_data
        otput_data_list = []
        for doy in doys:
            # data = _get_hls(doy)
            # croped_data = _crop_data(data, doy)
            tr_df = train_data[ train_data['DOY'] == int(doy)].copy()
            otput_data_list.append(tr_df)
            if SHOW == True:
                self._image_df(tr_df)
            # train_data_list.append(croped_data)
        # fn
        output_data = pd.concat(otput_data_list, axis=0)

        self.data = output_data.copy()

        self.data = self.data.reset_index(drop=True)

        return self.data

    def _set_columns_name(self, final_columns_list = ['B02', 'B03', 'B04', 'B05', 'NDVI', 'cloud', 'adj_cloud', 'cloud_shadow', 'X', 'Y', 'DOY', 'PID']):

        before_list = self.data.columns.to_list()
        
        self.data = self.data[final_columns_list]
        # train_data.columns = final_columns_list
        
        print(f'Change columns list: {before_list}->{self.data.columns.to_list()}')

        self.data = self.data.reset_index(drop=True)

        return self.data

    def _nan_9999(self, ):
        # Replace -9999 with np.nan
        # self.data.replace(-9999, np.nan, inplace=True)
        # Separate rows with NaN values and without NaN values
        # df_with_nan = 
        # df_without_nan = df[~df.isnull().any(axis=1)]
        self.nan_data = self.data[self.data.isnull().any(axis=1)].copy()
        # self.data = self.data[~self.data.isnull().any(axis=1)].copy()

        self.nan_data = self.nan_data.reset_index(drop=True)
        self.data = self.data.reset_index(drop=True)
        
        return self.data, self.nan_data

    ###### GET CLEAR PIXELS ONLY ###########################
    def _set_clear_cloud(self, ):
        # Remove rows containing np.nan in any column
        train_cleaned = self.data.dropna(how='any').copy()
        self.clear_data = train_cleaned.loc[(train_cleaned[f'cloud'] == 0) & \
                                            (train_cleaned[f'adj_cloud'] == 0) & \
                                            (train_cleaned[f'cloud_shadow'] == 0)].copy()

        self.cloud_data = train_cleaned.loc[(train_cleaned[f'cloud'] == 1) | \
                                            (train_cleaned[f'adj_cloud'] == 1) | \
                                            (train_cleaned[f'cloud_shadow'] == 1)].copy()

        self.clear_data = self.clear_data.reset_index(drop=True)
        self.cloud_data = self.cloud_data.reset_index(drop=True)
        
        return self.clear_data, self.cloud_data

    def _set_train_columns_name(self, final_columns_list = ['B02', 'B03', 'B04', 'B05', 'NDVI', 'X', 'Y', 'DOY', 'PID']):

        before_list = self.data.columns.to_list()
        
        self.data = self.data[final_columns_list].astype(self.table_dtype, errors='ignore')
        self.nan_data = self.nan_data[final_columns_list].astype(self.table_dtype, errors='ignore')
        self.clear_data = self.clear_data[final_columns_list].astype(self.table_dtype, errors='ignore')
        self.cloud_data = self.cloud_data[final_columns_list].astype(self.table_dtype, errors='ignore')
        # train_data.columns = final_columns_list
        
        print(f'Change columns list: {before_list}->{self.data.columns.to_list()}')

        self.data = self.data.reset_index(drop=True)
        self.nan_data = self.nan_data.reset_index(drop=True)
        self.clear_data = self.clear_data.reset_index(drop=True)
        self.cloud_data = self.cloud_data.reset_index(drop=True)
        
        return self.data, self.nan_data, self.clear_data, self.cloud_data

    def _set_train_test_data(self, doy, x1, y1, x2, y2, for_show_nan=False):

        x_shift = self.clear_data['X'].min()
        y_shift = self.clear_data['Y'].min()
        
        self.test_data = self.clear_data.loc[(self.clear_data['DOY'] == doy) & \
                                             (self.clear_data['X'] >= x1 + x_shift) & \
                                             (self.clear_data['X'] <= x2 + x_shift) & \
                                             (self.clear_data['Y'] >= y1 + y_shift) & \
                                             (self.clear_data['Y'] <= y2 + y_shift)].copy()
        # #### check test box, comment it ###########
        if for_show_nan == True:
            self.test_data.loc[:,['B02', 'B03', 'B04', 'B05', 'NDVI']] = np.NaN

        self.train_data = self.clear_data.loc[(self.clear_data['DOY'] != doy) | \
                                             ((self.clear_data['X'] < x1 + x_shift) | \
                                              (self.clear_data['X'] > x2 + x_shift) | \
                                              (self.clear_data['Y'] < y1 + y_shift) | \
                                              (self.clear_data['Y'] > y2 + y_shift))].copy()
                                              
        self.test_data = self.test_data.reset_index(drop=True)              
        self.train_data = self.train_data.reset_index(drop=True)   

        return self.train_data, self.test_data

    def _set_timeseries_train_test_data(self, doy, x1, y1, x2, y2):

        x_shift = self.clear_data['X'].min()
        y_shift = self.clear_data['Y'].min()
        
        self.test_data = self.clear_data.loc[(self.clear_data['X'] >= x1 + x_shift) & \
                                             (self.clear_data['X'] <= x2 + x_shift) & \
                                             (self.clear_data['Y'] >= y1 + y_shift) & \
                                             (self.clear_data['Y'] <= y2 + y_shift)].copy()
        # #### check test box, comment it ###########
        self.test_data.loc[:,['B02', 'B03', 'B04', 'B05', 'NDVI']] = np.NaN

        self.train_data = self.clear_data.loc[(self.clear_data['X'] < x1 + x_shift) | \
                                              (self.clear_data['X'] > x2 + x_shift) | \
                                              (self.clear_data['Y'] < y1 + y_shift) | \
                                              (self.clear_data['Y'] > y2 + y_shift)].copy()
                                              
        self.test_data = self.test_data.reset_index(drop=True)              
        self.train_data = self.train_data.reset_index(drop=True)                                         

        return self.train_data, self.test_data

    def _to_impute(self,):
        self.to_impute = self.cloud_data.copy()
        # display(self.impute_data)
        self.to_impute.loc[:,['B02', 'B03', 'B04', 'B05', 'NDVI']] = np.NaN

        self.to_impute = pd.concat([self.to_impute, self.nan_data], axis=0).copy()
        self.imputed_data = self.to_impute
        
        print('to_impute:')
        display(self.to_impute)

        return self.to_impute

    def _imputed_data(self, data):
        self.imputed_data = data.copy()

        return self.imputed_data

    def _inference_train_test_data(self,):
        self.inference_data = pd.concat([self.imputed_data, self.train_data, self.test_data], axis=0)
        # Sort the DataFrame by 'X', 'Y', and 'DOY'
        self.inference_data = self.inference_data.sort_values(by=['Y', 'X', 'DOY', ])
        # test_data = test_data.sort_values(by=['Y', 'X', 'DOY'])
        return self.inference_data
        
    def _inference_clear_data(self,):
        self.inference_data = pd.concat([self.imputed_data, self.clear_data], axis=0)
        # Sort the DataFrame by 'X', 'Y', and 'DOY'
        self.inference_data = self.inference_data.sort_values(by=['Y', 'X', 'DOY', ])
        # test_data = test_data.sort_values(by=['Y', 'X', 'DOY'])
        return self.inference_data

    def _inference_imshow(self, filename='inference.jpg'):
        doys = self.inference_data['DOY'].unique()

        # self.impute_data = pd.concat([self.impute_data, self.nan_data], axis=0)
        test_data = self.inference_data
        orig_data = self.data
        
        for doy in doys:
            # data = _get_hls(doy)
            # croped_data = _crop_data(data, doy)
            tr_df  = test_data[ test_data['DOY'] == int(doy)].copy()
            tr_df2 = orig_data[ orig_data['DOY'] == int(doy)].copy()
            # otput_data_list.append(tr_df)
            self._image_inference(tr_df2, tr_df, filename)

    def _image_inference(self, input1, input2, filename):
        # BND_LIST = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B09', 'B10', 'B11']
        # FMSK_LIST = ['cirrus', 'cloud', 'adj_cloud', 'cloud_shadow', 'snow_ice', 'water', 'aero' ]
        # ANG_LIST = ['SAA', 'SZA', 'VAA', 'VZA',]
    
        input = input1
    
        box_x_size = (input['X'].max() - input['X'].min() + 1).astype('int')
        box_y_size = (input['Y'].max() - input['Y'].min() + 1).astype('int')
    
        def _get_img_nan(input, bnd_list=['B04', 'B03', 'B02']):
    
            print(input['DOY'].unique())
    
            df = input[bnd_list].copy()
    
            df[df > 0] = 0
            df[df == -9999] = 1
            df[df < 0] = 0
    
            image = df.to_numpy()  # df[chanel_list]
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            nans = np.dstack((image[0,:,:], image[1,:,:], image[2,:,:]))
    
            return nans
    
    
        def _get_img_rgb(input, bnd_list=['B04', 'B03', 'B02']):
            df = input[bnd_list].copy()
    
            # df[df > 0] = 0
            df[df == -9999] = np.nan
            # df[df < 0] = 0
    
            image = df.to_numpy()
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            # Convert the int16 array to int64
            # image = image.astype(np.uint64)
    
            r_ = (8200, 16000)
            g_ = (8500, 14000)
            b_ = (7500, 12000)
    
    
            def generalized_normalization(band, rgb):
                # Apply your normalization method here
                # Example: Stretch and scale values to 0-255
                band = np.ma.array (band, mask=np.isnan(band))

                if self.REFLECT == False:
                ### FOR HLS #################
                    band = 0.0001 * band
                    band = np.where(band > 0.3, 0.3, band)
                    min_val = -0.063
                    max_val = 0.3
        
                    # min_val = np.min(band)
                    # max_val = np.max(band)
                    # min_val = rgb[0]
                    # max_val = rgb[1]
                    # print(f'gn:{min_val}, {max_val}')
                    normalized_band = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    normalized_band = band
                #
                # Replace elements greater than 2000 with 1
                # print('band:', band.min(), band.max())
                normalized_band[normalized_band == np.nan] = 255
                normalized_band = normalized_band.astype(np.uint8)
                return normalized_band
    
            # Scale the bands to 8-bit
            scaled_red = generalized_normalization(image[2,:,:], r_)
            scaled_green = generalized_normalization(image[1,:,:], g_)
            scaled_blue = generalized_normalization(image[0,:,:], b_)
    
            rgb = np.dstack((scaled_red, scaled_green, scaled_blue))
    
            return rgb

        def _get_img_ndvi(input, bnd_list=['NDVI']):
            df = input[bnd_list].copy()
    
            # df[df > 0] = 0
            df[df == -9999] = np.nan
            # df[df < 0] = 0
    
            image = df.to_numpy()
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            # Convert the int16 array to int64
            # image = image.astype(np.uint64)
    
            r_ = (8200, 16000)
            g_ = (8500, 14000)
            b_ = (7500, 12000)
    
    
            def ndvi_normalization(band):
                # Apply your normalization method here
                # Example: Stretch and scale values to 0-255
                band = np.ma.array (band, mask=np.isnan(band))
    
                ### FOR HLS #################
                # band = 0.0001 * band
                # band = np.where(band > 0.3, 0.3, band)
                min_val = 0.0
                max_val = 1.0
    
                normalized_band = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                normalized_band[normalized_band == np.nan] = 255
                return normalized_band
    
            # Scale the bands to 8-bit
            scaled_ndvi = ndvi_normalization(image[0,:,:])
    
            return scaled_ndvi
    
        def _get_img(input, bnd_list):
            image = input[bnd_list].to_numpy()  # df[chanel_list]
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
    
            # Convert the int16 array to int64
            image = image.astype(np.uint64)
    
            return image[0,:,:]

        def _get_hist(input, bnd_list):
            image = input[bnd_list].to_numpy()  # df[chanel_list]
    
            image = image.transpose()
            image = image.reshape(image.shape[0], box_x_size, box_y_size)
                     
            image_plt = image  # (image * 255).astype(np.uint8)

            print('image:', image.shape, image)
            # fn

            ############################################
            # Check for NaN values
            nan_mask = np.isnan(image)
            # Calculate the average of non-NaN values
            average_value = np.nanmax(image)        
            # Replace NaN values with the average
            image_plt = image.copy()
            image_plt[nan_mask] = average_value
            # fn

            image_plt = image_plt.astype(np.uint8)

            ####################################################
            # Replace NaN values with the mean of non-NaN values
            # mean_value = np.nanmean(image)
            # image[np.isnan(image)] = mean_value     
            # Normalize the array using z-score normalization
            hist = image # (image - np.mean(image)) / np.std(image)

    
            # Convert the int16 array to int64
            # image_plt = image_plt.astype(np.uint64)
    
            
            fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
            ax[0].imshow(image_plt[0,:,:], cmap=plt.cm.gray)
            ax[0].axis("off")
            ax[0].set_title("Rendering of the image")
            ax[1].hist(hist[0,:,:].ravel(), bins=256)
            ax[1].set_xlabel("Pixel value")
            ax[1].set_ylabel("Count of pixels")
            ax[1].set_title("Distribution of the pixel values")
            _ = fig.suptitle("Original image of a hls dataset")

            return image[0,:,:]
        # fn

        hist = _get_hist(input2, bnd_list=['B03'])
    
        # image_nan = _get_img_nan(input, bnd_list=['B02', 'B03', 'B04'])
        image_rgb   = _get_img_rgb(input1, bnd_list=['B02', 'B03', 'B04'])
    
        image_rgb2  = _get_img_rgb(input2, bnd_list=['B02', 'B03', 'B04'])

        image_band   = _get_img_rgb(input1, bnd_list=['B02', 'B03', 'B04'])
    
        image_band2  = _get_img_rgb(input2, bnd_list=['B02', 'B03', 'B04'])

        # image_ndvi  = _get_img_ndvi(input1, bnd_list=['NDVI'])
    
        # image_ndvi2 = _get_img_ndvi(input2, bnd_list=['NDVI'])
    
        # image_rgb_list = [image_rgb, image_ndvi, image_rgb2, image_ndvi2]
        image_rgb_list = [image_rgb, image_band, image_rgb2, image_band2]
    
        # Create subplots
        fig, axes = plt.subplots(2, 2)  #, figsize=(9, 11))    # (1, len(image_rgb_list))    # , figsize=(18, 22))
        # Flatten the axes array to simplify indexing
        axes = axes.ravel()
        # print(image_rgb_list[0].shape)
        axes[0].imshow(image_rgb_list[0])
        axes[1].imshow(image_rgb_list[1], cmap=plt.cm.summer)
        axes[2].imshow(image_rgb_list[2])
        axes[3].imshow(image_rgb_list[3], cmap=plt.cm.summer)
        # # Loop through the images and plot them
        # for ii in range(1,len(image_rgb_list)):
        #     axes[ii].imshow(image_rgb_list[ii]) #, cmap='gray')  # You can specify a colormap
        plt.tight_layout()
        plt.show()

        plt.savefig(filename)

        return filename

