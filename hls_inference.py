from hlsdataset import HLSDataSet

import numpy as np

import pandas as pd

class HLSInference(HLSDataSet):
    def __init__(self, doy=211, table_dtype = 'float16', path='./aispace/data/L8-100x100'):
        super().__init__(table_dtype = table_dtype, path=path)
        # self.model = model
        self.doy_to_impute = doy

        self.imputed_data = None
        self.recovered_data = None

        self.imputed_file = ''

        # self.nan_data = self.to_impute #.copy()    # to_impute.copy()
        # display(self.nan_data)
        #### for only one day processing ###########################
        self.nan_data_doy = None # self.nan_data.loc[(self.nan_data['DOY'] == doy)]
        self.nan_data_doy = None #  self.nan_data_doy.reset_index(drop=True)

        self.data_doy = None #  self.clear_data.loc[(self.clear_data['DOY'] == doy)]
        self.data_doy = None #  self.data_doy.reset_index(drop=True)

        # self.nan_data_resid  = self.nan_data.loc[(nan_data['DOY'] != doy)]
        # self.nan_data_resid  = self.nan_data_resid.reset_index(drop=True)

    def _impute(self, model, columns_impute=['PID', 'DOY', 'B02', 'B03', 'B04'], device='cuda', k=400, max_length=1000, temperature=0.01):
    # def _impute(self, model, columns_impute=['B02', 'B03', 'B04', 'X', 'Y', 'DOY'], device='cuda', k=400, max_length=1000, temperature=0.01):

        self.nan_data_doy = self.to_impute.loc[(self.to_impute['DOY'] == self.doy_to_impute)]
        self.nan_data_doy = self.nan_data_doy.reset_index(drop=True)

        self.data_doy = self.clear_data.loc[(self.clear_data['DOY'] == self.doy_to_impute)]
        self.data_doy = self.data_doy.reset_index(drop=True)

        # print(f'to impute data:')
        # display(self.nan_data_doy)
        # ########### PRINT ###################################
        # print(f'to impute data:')
        # display(self.nan_data_doy)
        # display(self.data_doy)

        # fn         

        # print(f'NumPy version:{np.__version__}')
        np.float = float

        # Get the DataFrame with columns in reverse order
        # self.nan_data_doy = self.nan_data_doy[self.nan_data_doy.columns[::-1]].copy()

        ######################### ORIGINAL ##############################################################
        _impute = self.nan_data_doy[columns_impute].copy()
        # _impute[['DOY', 'PID']] = _impute[['DOY', 'PID']].astype(int)


        # ################# PRINT ##########################
        # print('IMPUTE:')
        # display(_impute)
        
        # self.imputed_data = pd.read_csv('imputed.csv')
        # self.imputed_data = model.impute(_impute, k=k, max_length=max_length, temperature=temperature, device=device)
        self.imputed_data = model(_impute, k=k, max_length=max_length, temperature=temperature, device=device)
        self.imputed_data.to_csv('imputed2.csv')

        # #### PRINT ############################
        # print('IMPUTED:')
        # display(self.imputed_data)
        
        #########################################################################################
        # ################################## FOR BANDS in ONE BAND#####################################################
        # # _impute = self.nan_data_doy[columns_impute].copy()
        # _impute = self.nan_data_doy[columns_impute].copy()
        # _impute[['PID', 'DOY', 'BAND']] = self.nan_data_doy[['PID', 'DOY', 'B03']].copy()
        # _impute = _impute[['PID', 'DOY', 'BAND']].copy()
        # print('IMPUTE:')
        # display(_impute)
        # # self.imputed_data = pd.read_csv('imputed.csv')
        # self.imputed_data = model.impute(_impute, k=k, max_length=max_length, temperature=temperature, device=device)
        # self.imputed_data.to_csv('imputed2.csv')

        # print('IMPUTED:')
        # display(self.imputed_data)

        # # # Inverse operation: Split 'BAND' column back into individual columns
        # aa = self.nan_data_doy[columns_impute].copy()
        # display(aa)
        
        # # aa[['B02', 'B03', 'B04', 'NN']]  
        # aa = self.imputed_data['BAND'].str.split(';', expand=True)
        # display(aa)
        
        # self.imputed_data[['B02', 'B03', 'B04', 'NN']] = self.imputed_data['BAND'].str.split(';', expand=True)

        # # # Convert columns back to their original data types
        # self.imputed_data[['B02', 'B03', 'B04']] = self.imputed_data[['B02', 'B03', 'B04']].astype(table_dtype)

        # print('IMPUTED:')
        # display(self.imputed_data)
        # #########################################################################################
        
        # imputed_file = f'A0[optim_sophia]/imputed_output_run[3].csv'
        # self.imputed_data = pd.read_csv(imputed_file)
        # self.imputed_data = self.nan_data_doy.copy()

        # #### PRINT TO IMPUTE DATA #######################################
        # print(self.nan_data_doy.columns, self.imputed_data.columns)
        # print(self.nan_data_doy.shape, self.imputed_data.shape)

        # Merge the dataframes by X and Y columns and replace B3 in df1 with B3 from df2
        merged_df = self.nan_data_doy.merge(self.imputed_data, on=['PID', 'DOY'], suffixes=('', '_df2'), how='left')  
        # Replace the original B3 column with B3 from df2
        merged_df['B02'] = merged_df['B02_df2']
        merged_df['B03'] = merged_df['B03_df2']
        merged_df['B04'] = merged_df['B04_df2']
        
        # Drop the additional B3_df2 column
        merged_df = merged_df.drop('B02_df2', axis=1)
        merged_df = merged_df.drop('B03_df2', axis=1)
        merged_df = merged_df.drop('B04_df2', axis=1)

        self.imputed_data = merged_df.copy()

        # #### PRINT IMPUTED DATA #######################################
        # print(self.nan_data_doy.columns, self.imputed_data.columns)
        # print(self.nan_data_doy.shape, self.imputed_data.shape)
        # fn
        # cols = self.nan_data_doy.columns 
        # self.imputed_data

        # Get the DataFrame with columns in reverse order
        self.imputed_data = self.imputed_data[self.imputed_data.columns[::-1]].copy()

        if len(self.imputed_data) != len(self.nan_data_doy):
            print('len(self.imputed_data) != len(self.nan_data_doy)')
            # Use the merge function with indicator=True
            original_df = self.nan_data_doy
            subset_df = self.imputed_data

            merged_df = pd.merge(original_df, subset_df, on=['PID', 'DOY'], how='left', indicator=True)

            # Find the rows in original_df that are not in subset_df
            missing_rows = original_df[merged_df['_merge'] == 'left_only']

            # Display the missing rows
            # display(missing_rows)

            self.imputed_data = pd.concat([self.imputed_data, missing_rows], axis=0)

        self.recovered_data = pd.concat([self.imputed_data, self.data_doy], axis=0)
        self.recovered_data = self.recovered_data.reset_index(drop=True)

        # #### PRINT ###############################
        # print('imputed_data')
        # display(self.imputed_data)

        # print('recovered_data')
        # display(self.recovered_data)

        return self.recovered_data

    def _set_inference_recovered(self,):
       self.inference_data = self.recovered_data
       self.inference_data = self.inference_data.sort_values(by=['Y', 'X', 'DOY', ])

       return self.inference_data

    def _save_recovered(self, imputed_file=f'recovered_output.csv'):
        self.imputed_file = imputed_file
        print(imputed_file)
        self.recovered_data.to_csv(self.imputed_file)

    def _read_recovered(self, imputed_file=f'recovered_output.csv'):
        self.imputed_file = imputed_file
        print(imputed_file)
        self.recovered_data = pd.read_csv(self.imputed_file)
        # display(self.imputed_data)
        # ######### CLEAR UNNAMED COLUMNS FROM DATASETS #######################################
        self.recovered_data = self.recovered_data.loc[:, ~self.recovered_data.columns.str.contains('^Unnamed')]
        return self.recovered_data

    def _save_imputed(self, imputed_file=f'imputed_output.csv'):
        self.imputed_file = imputed_file
        print(imputed_file)
        self.imputed_data.to_csv(self.imputed_file)

    def _read_imputed(self, imputed_file=f'imputed_output.csv'):
        self.imputed_file = imputed_file
        print(imputed_file)
        self.imputed_data = pd.read_csv(self.imputed_file)
        # display(self.imputed_data)
        # ######### CLEAR UNNAMED COLUMNS FROM DATASETS #######################################
        self.imputed_data = self.recovered_data.loc[:, ~self.recovered_data.columns.str.contains('^Unnamed')]
        return self.imputed_data