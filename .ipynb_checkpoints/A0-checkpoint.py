import numpy as np

import great
from great import GReaT

import pandas as pd

################################
import numpy as np

import pandas as pd

# all imports should go here

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show

import sklearn
from sklearn.model_selection import train_test_split

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

from sklearn.model_selection import train_test_split

from hlsdataset import HLSDataSet

import shutil

import wandb

from multiprocessing import Pool, freeze_support
import warnings

import fire

# Execute only once!
import os
import sys
sys.path.append("..")
os.chdir("..")

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

class BaseModel:
    def __init__(self, EXP_NAME=f'train_ai'):
        self.EXP_NAME = EXP_NAME
####################### LOAD AND VISUALIZE DATASET#############################################################
        # Get the current working directory
        current_directory = os.getcwd()
        print(current_directory)
        
        print(f'NumPy version:{np.__version__}')
        np.float = float # 'float32' # float
        table_dtype = int  # 'int64'   # np.float  #'float32'
        print(table_dtype)
        
        hls_data = HLSDataSet(table_dtype = table_dtype)
        
        # hls_data._REFLECTANCE(round=3)
        # hls_data._QUANTIZATE(round=3)
        
        #### clip 50x50 subdataset
        # hls_data.clip_dataset(x1=50.0, y1=50.0, x2=100.0, y2=100.0)
        # doys = [171, 179, 187, 195, 203, 211, 219]
        # doys = [171, 179, 187, 203, 211, 219]
        doys = [171, 187, 203, 211, 219]
        # doys = [203, 211, 219]
        # doys = [203, 211, 219]
        
        df = hls_data._get_data_doys(doys = doys, SHOW=True)
        print(df)
        
        # df = hls_data._set_columns_name()
        # display(df)
        
        df1, df2 = hls_data._nan_9999()
        # display(df1)
        # display(df2)
        
        df1, df2 = hls_data._set_clear_cloud()
        # display(df1)
        # display(df2)
        
        data, nan, clear, cloud = hls_data._set_train_columns_name()
        
        print('clear')
        print(clear)
        
        # train_data, test_data = hls_data._set_train_test_data(doy=211.0, x1=60.0, y1=40.0, x2=75.0, y2=55.0)
        
        train_data, test_data = hls_data._set_train_test_data(doy=211.0, x1=45.0, y1=45.0, x2=50.0, y2=50.0)
        
        # train_data, test_data = hls_data._set_train_test_data(doy=211.0, x1=95.0, y1=0.0, x2=100.0, y2=5.0, for_show_nan=False)
        
        # train_data, test_data = hls_data._set_train_test_data(doy=187.0, x1=2.0, y1=0.0, x2=21.0, y2=50.0, for_show_nan=False)
        # train_data, test_data = hls_data._set_timeseries_train_test_data(doy=211.0, x1=50.0, y1=50.0, x2=100.0, y2=100.0)
        
        print('train_data:')
        print(train_data)
        print('test_data:')
        print(test_data)
        # display(data)
        # display(nan)
        # display(clear)
        # display(cloud)
        
        hls_data._to_impute()
        hls_data._inference_train_test_data()
        hls_data._inference_imshow()
####################################################################################
######################## SET TRAINING DATASET ######################################
        self.train_columns_list = ['DOY', 'PID', 'B02', 'B03', 'B04'] 
        
        self.data = train_data[self.strain_columns_list]
        img_data = test_data[self.train_columns_list]
        
        print(self.data)
####################################################################################
######################## ReTRAIN TOKENIZER FOR HLS DATA ############################
    def _ReTokinizer(self, ):
        # #### TRAIN ONCE AND COMMENT FOR AVOID FORKING TOKENIZER ##########################
        
        # EXP_NAME = f'A0[RGB_TOK[ai]_Lora]'
        # EXP_NAME = f'A0[RGB_REFL[int]BAND]'
        # EXP_NAME = f'A0[RGB_REFL[int]3BAND]'
        
        # self.EXP_NAME = f'A0[RGB_ORIG[int]3BAND]'
        
        from huggingface_hub import notebook_login
        
        # notebook_login()
        
        ReTRAIN_TOKENIZER = True
        if ReTRAIN_TOKENIZER == True:
            def _get_synth_data():
                BAND = [b for b in range(0, 10)]
                PID  = [p for p in range(0, 10)]
                DOY  = [d for d in range(0, 10)]   
                # BNAME = ['B02', 'B03', 'B04']
                # Create a DataFrame with specified column names
                # synth_data = pd.DataFrame(list(zip(BAND, PID, DOY)), columns=['BAND', 'PID', 'DOY'])
                synth_data = pd.DataFrame(list(zip(BAND, BAND, BAND, PID, DOY)), columns=['B02', 'B03', 'B04', 'PID', 'DOY'])
                # synth_data = pd.DataFrame(list(zip(BAND, BAND, BAND, PID, PID, DOY)), columns=['B02', 'B03', 'B04', 'X', 'Y', 'DOY'])  
        
                # # Create a list of tuples with all possible combinations of DOY and PID
                # synth_data = [(d, p, b) for d in DOY for p in PID for b in BAND]
                # # Create a DataFrame from the data
                # synth_data = pd.DataFrame(synth_data, columns=["DOY", "PID", "B03"])
                return synth_data.astype(table_dtype)
            
            synth_data = _get_synth_data()
        
            synth_data = self.data
            
            # Display the DataFrame
            display(synth_data)
            
            from transformers import AutoTokenizer
            from aitokenizer import TrainTokenizer, TrainCorpus, PreTrainTokenizer
            
            old_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            old_tokenizer.pad_token = old_tokenizer.eos_token
            
            data_corpus = TrainCorpus(synth_data)._get_training_corpus()
        
            sample = next(data_corpus)[0]
            print('L8 data:', sample)
            
            tokens = old_tokenizer.tokenize(sample)
            print('old_tokenizer:', len(tokens), tokens)
            # print(old_tokenizer(sample).tokens())
        
            digits_tok  = [str(x) for x in range(0,10)]
            # digits_tok2 = [' ' + str(x) for x in range(0,10)]
            
            new_tokenizer = PreTrainTokenizer(self.data, old_tokenizer)._train(special_tokens=digits_tok+self.train_columns_list+[';',])
            
            # print(new_tokenizer(sample).tokens())
            tokens = new_tokenizer.tokenize(sample)
            print('aispace-tokenizer:', len(tokens), tokens)
            
            new_tokenizer.save_pretrained(f"{self.EXP_NAME}/aispace-tokenizer")
        
            print('Tokenizer trained, COMMENT & RELOAD Tokenizer Trainer...')
            sys.exit(0)
        ################################################################################################

####################################################################################
class RunTask:
    @staticmethod
    def train_ai(EXP_NAME=f'train_ai'):

        # EXP_NAME=f'A0[RGB_ORIG[int]3BAND]'
        # EXP_NAME=f'train_ai'

        model = BaseModel(EXP_NAME)

        model._ReTokenizer()
        
        print('The end...')
        sys.exit(0)

if __name__ == "__main__":
    
    freeze_support()
    warnings.filterwarnings("ignore")
    
    fire.Fire(RunTask)

