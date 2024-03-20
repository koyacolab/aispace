###### IMPORT ################

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

class ModelBase:

    def __init__(self,
                 EXP_NAME = f'A0[VIRTUAL_SAT]',                
                 save_steps = 100000,
                 logging_steps = 1000, 
                 TRAINER_RUN = 0,
                 batch_size_list = [8, 8, 8, 8],
                 max_steps_list = [2500000, 400000, 2200000, 400000, 400000],
                 warmup_steps_list = [ 500000, 200000,  200000, 200000, 200000],
                 lr_scheduler_type_list = ['cosine', 'cosine', 'cosine', 'cosine'],
                 num_cycles_list = [5, 5, 5, 5],
                 learning_rate_list = [2e-6, 2e-6, 2e-5, 2e-6, 2e-6],   
                 step_checkpoint_list = [400000, 400000, 400000, 400000],     
                 RESUME_FROM_CHECKPOINT = False,
                 report_to = 'none',
                ):

        ################################################################################
        
        self.MODEL_NAME = 'distilgpt2'
        # MODEL_NAME = 'gpt2-large'
        
        ################################################################################

        self.batch_size_list = batch_size_list

        self.save_steps  = save_steps

        self.logging_steps = logging_steps

        self.TRAINER_RUN = TRAINER_RUN

        self.max_steps_list = max_steps_list

        self.warmup_steps_list = warmup_steps_list

        self.lr_scheduler_type_list = lr_scheduler_type_list

        self.num_cycles_list = num_cycles_list

        self.learning_rate_list = learning_rate_list  

        self.step_checkpoint_list = step_checkpoint_list

        self.RESUME_FROM_CHECKPOINT = RESUME_FROM_CHECKPOINT

        self.report_to = report_to
        
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
        
        # Get the current working directory
        current_directory = os.getcwd()
        print(current_directory)
        
        print(f'NumPy version:{np.__version__}')
        np.float = float # 'float32' # float
        self.table_dtype = int  # 'int64'   # np.float  #'float32'
        print(self.table_dtype)
        
        hls_data = HLSDataSet(table_dtype = self.table_dtype)
        
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
        # display(df)
        
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
        # display(clear)
        
        # train_data, test_data = hls_data._set_train_test_data(doy=211.0, x1=60.0, y1=40.0, x2=75.0, y2=55.0)
        
        self.train_data, self.test_data = hls_data._set_train_test_data(doy=211.0, x1=45.0, y1=45.0, x2=50.0, y2=50.0)
        
        # train_data, test_data = hls_data._set_train_test_data(doy=211.0, x1=95.0, y1=0.0, x2=100.0, y2=5.0, for_show_nan=False)
        
        # train_data, test_data = hls_data._set_train_test_data(doy=187.0, x1=2.0, y1=0.0, x2=21.0, y2=50.0, for_show_nan=False)
        # train_data, test_data = hls_data._set_timeseries_train_test_data(doy=211.0, x1=50.0, y1=50.0, x2=100.0, y2=100.0)
        
        print('train_data:')
        # display(self.train_data)
        print('test_data:')
        # display(self.test_data)
        # display(data)
        # display(nan)
        # display(clear)
        # display(cloud)
        
        hls_data._to_impute()
        hls_data._inference_train_test_data()
        hls_data._inference_imshow()

        self.EXP_NAME = EXP_NAME

        ######### FOR 3BAND ##############################################
        self.train_columns_list = ['PID', 'DOY', 'B02', 'B03', 'B04'] 
        
        self.data = self.train_data[self.train_columns_list]
        img_data  = self.test_data[self.train_columns_list]
        
        # fn

    def Tokenize(self,):
    
        ########################################################
        
        from aitokenizer import _int_tiktoken, _tiktoken_int
        ###### SET MODEL #########################
        
        # MODEL_NAME = 'distilgpt2'
        # MODEL_NAME = 'gpt2-large'
        
        # #### TRAIN TOKENIZER FOR HLS DATA ################################################
        # #### TRAIN ONCE AND COMMENT FOR AVOID FORKING TOKENIZER ##########################
        
        # from huggingface_hub import notebook_login
        
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
                return synth_data.astype(self.table_dtype)
            
            synth_data = _get_synth_data()
        
            synth_data = self.data
            
            # Display the DataFrame
            # display(synth_data)
            
            from transformers import AutoTokenizer
            from aitokenizer import TrainTokenizer, TrainCorpus, PreTrainTokenizer
            
            old_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            old_tokenizer.pad_token = old_tokenizer.eos_token
            
            data_corpus = TrainCorpus(synth_data)._get_training_corpus()
        
            for ii in range(100):
                sample = next(data_corpus)[0]
            print('L8 data:', sample)
            
            tokens = old_tokenizer.tokenize(sample)
            print('old_tokenizer:', len(tokens), tokens)
            # print(old_tokenizer(sample).tokens())
        
            digits_tok = [str(x) for x in range(0,10)]
            # digits_tok2 = [' ' + str(x) for x in range(0,10)]
            int_digits_tiktok = ['_{:d}_{:d}_'.format(xx, yy) for xx in range(10) for yy in range(4)]
            # print(int_digits_tiktok[0:10])
            # fn
            
            # new_tokenizer = PreTrainTokenizer(data, old_tokenizer)._train(special_tokens=digits_tok+train_columns_list+[';',])
        
            new_tokenizer = old_tokenizer
            num_new_tokenizer = new_tokenizer.add_tokens(int_digits_tiktok+self.train_columns_list)
        
            # txt2 = _int_tokenize2(sample)
            # print('int_tokenized2:', txt2)
        
            sample = _int_tiktoken(sample)
            print('int_tiktoken:', sample)
        
            int_sample = _tiktoken_int(sample)
            print('tiktoken_int:', int_sample)
        
            # sample = txt2
        
            # fn
        
            # num_new_tokens = tokenizer.add_special_tokens(special_tokens)
            # vocab = tokenizer.get_vocab()
            # model.resize_token_embeddings(len(vocab))
            
            # print(new_tokenizer(sample).tokens())
            tokens = new_tokenizer.tokenize(sample)
            print('aispace-tokenizer:', len(tokens), tokens)
            
            new_tokenizer.save_pretrained(f"{self.EXP_NAME}/aispace-tokenizer")
        
            print('Tokenizer trained, COMMENT & RELOAD Tokenizer Trainer...')

            sys.exit(0)
        
            fn
    
    ################################################################################################

    def train(self,):
    
        #### LISTS for TRAINER_RUN cycles COSINE WITH WARM UP ###################################
        epochs_list = [100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 

        num_cycles = self.num_cycles_list[self.TRAINER_RUN]    #### 
        
        # os.environ["WANDB_SILENT"] = "true"
        os.environ['HF_EVALUATE_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'True'
        
        # wandb.login(relogin=True, force=True, key='10a24a9f6330ab3c52f92ce91d927df68c6058ac')  #--relogin
        
        #### SET MODEL LOADING FROM: 0 - 'distilgpt2', else - checkpoint-'' 
        if self.TRAINER_RUN == 0:
            load_model_path = self.MODEL_NAME
            experiment_dir = f"{self.EXP_NAME}/run[{self.TRAINER_RUN}]"
            if self.RESUME_FROM_CHECKPOINT == True:
                step_checkpoint = self.step_checkpoint_list[self.TRAINER_RUN]
                # load_model_path = self.MODEL_NAME
                # load_model_path = f'{self.EXP_NAME}/run[{self.TRAINER_RUN}]' + f'/checkpoint-{step_checkpoint}'
                print(f'Resume from checkpoint:{load_model_path}')
        elif self.TRAINER_RUN > 0:
            # before = pd.read_csv(f'{self.EXP_NAME}/run[{self.TRAINER_RUN-1}].csv')
            # load_model_path = before['experiment_dir'][0] + f'/checkpoint-{step_checkpoint}'
            if self.RESUME_FROM_CHECKPOINT == True:
                step_checkpoint = self.step_checkpoint_list[self.TRAINER_RUN]
                load_model_path = f'{self.EXP_NAME}/run[{self.TRAINER_RUN}]' + f'/checkpoint-{step_checkpoint}'
                print(f'Resume from checkpoint:{load_model_path}')
            else:
                step_checkpoint =self.step_checkpoint_list[self.TRAINER_RUN-1]
                load_model_path = f'{self.EXP_NAME}/run[{self.TRAINER_RUN-1}]' + f'/checkpoint-{step_checkpoint}'
            experiment_dir  = f"{self.EXP_NAME}/run[{self.TRAINER_RUN}]"
        
        ###########################################################
        #### TRAINER HYPERPARAMETERS #############################
        
        epochs_list = [100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 

        batch_size = self.batch_size_list[self.TRAINER_RUN]

        num_cycles = self.num_cycles_list[self.TRAINER_RUN] 
        
        epochs = epochs_list[self.TRAINER_RUN]    # ignored, use max_steps  
        
        max_steps = self.max_steps_list[self.TRAINER_RUN] # if -1 max_steps ignored 
        
        learning_rate = self.learning_rate_list[self.TRAINER_RUN]
        
        lr_scheduler_type = self.lr_scheduler_type_list[self.TRAINER_RUN]
        
        warmup_steps = self.warmup_steps_list[self.TRAINER_RUN]

        # report_to = self.report_to
        
        optimizer = 'sophia'
        # optimizer = 'adamw_torch'
        #######################################################################
        ##### SET experiment_dir & efficient_finetuning #######################
        efficient_finetuning = ''  #'lora'
        #######################################################################
        if self.report_to == 'wandb':
            ##### WANDB CONFIG ###################################################
            ## 🐝 1️⃣ Start a new run to track this script
            wandb.init(
                      # Set the project where this run will be logged
                      project = "aispace", 
                      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                      name = f'{self.EXP_NAME}/run[{self.TRAINER_RUN}]', 
                      reinit = True,
                      force = True,
                      tags = [f'lr={str(self.learning_rate_list[self.TRAINER_RUN])}', 
                              f'optim={optimizer}',
                              f'scheduler={str(self.lr_scheduler_type_list[self.TRAINER_RUN])}', 
                              f'max_steps={str(self.max_steps_list[self.TRAINER_RUN])}', 
                              f'warmup={str(self.warmup_steps_list[self.TRAINER_RUN])}', 
                              f'MODEL_NAME={self.MODEL_NAME}'], 
                      notes = 'tokenizer: distilgpt2',
                      # magic=True,
                      # Track hyperparameters and run metadata
                      # config={
                      # # "learning_rate": learning_rate,
                      # "architecture": "GPT2",
                      # "dataset": "HLS/L-8",
                      # # "max_steps": max_steps,
                      # })   
                      )
        else:
            os.environ['WANDB_DISABLED'] = 'true'
            ######################################################################
        
        TRAINER_DICT = pd.DataFrame({'EXP_NAME' : self.EXP_NAME}, index = [self.TRAINER_RUN])    # dict()
        TRAINER_DICT['TRAINER_RUN'] = self.TRAINER_RUN
        TRAINER_DICT['table_dtype'] = self.table_dtype
        TRAINER_DICT['experiment_dir'] = experiment_dir
        TRAINER_DICT['save_steps'] = self.save_steps
        TRAINER_DICT['logging_steps'] = self.logging_steps
        TRAINER_DICT['epochs'] = epochs
        TRAINER_DICT['max_steps'] = max_steps
        TRAINER_DICT['batch_size'] = batch_size
        TRAINER_DICT['learning_rate'] = learning_rate
        TRAINER_DICT['lr_scheduler_type'] = lr_scheduler_type
        TRAINER_DICT['warmup_steps'] = warmup_steps
        TRAINER_DICT['num_cycles'] = num_cycles
        TRAINER_DICT['optimizer'] = optimizer
        TRAINER_DICT['fp16'] = True
        
        print(pd.DataFrame.from_dict(TRAINER_DICT))
        
        print('experiment_dir :', experiment_dir)
        print('load_model_path:', load_model_path)
        
        # Get the current CPU time in seconds since the epoch
        current_time = int(time.time())
        # Use the current time as a seed for a random number generator
        random_seed_state = current_time  # You can use this random_state for various random processes
        
        model = GReaT(llm=load_model_path,
                      # tokenizer=load_model_path,
                      tokenizer = f"{self.EXP_NAME}/aispace-tokenizer",      
                      # tokenizer = self.MODEL_NAME,     #### for distilgpt2 origin tokenizer 
                      # auto_find_batch_size=True,
                      batch_size = batch_size, epochs = epochs, max_steps = max_steps,
                      logging_steps = self.logging_steps, save_steps = self.save_steps,
                      # evaluation_strategy='steps',
                      logging_first_step = True,
                      save_total_limit = 4,
                      # prediction_loss_only=True,
                      experiment_dir = experiment_dir,
                      dataloader_num_workers = 4,
                      efficient_finetuning = efficient_finetuning,
                      learning_rate = learning_rate,
                      lr_scheduler_type = lr_scheduler_type,
                      warmup_steps = warmup_steps,
                      num_cycles = num_cycles,
                      optimizer = optimizer,
                      # warmup_ratio=0.05,
                      # seed=1701849883,         #current_time,
                      # data_seed=3403699766,    #current_time+int(time.time()),
                      # optim=TRAINER_DICT['optimizer'],
                      fp16 = True,            #### comment   it for Ampere, for Volta disable it if 0 or nan in loss appears
                      # torch_compile=True,   #### uncomment it for Ampere
                      # bf16=True,            #### uncomment it for Ampere
                      report_to = self.report_to,
                      run_name = f'{self.EXP_NAME}/run[{self.TRAINER_RUN}]',
                      evaluation_strategy = "steps",
                      eval_steps = self.logging_steps,
                      label_names = self.train_columns_list,  # ["DOY", "PID", "B02", "B03", "B04"],    
                      # gradient_accumulation_steps=4,  
                      )
        
        # ############################ CHECK MODEL ARCHITECTURE ###########################################################
        # print(f'----------- Model architecture, efficient_finetuning: {efficient_finetuning} -----------------------')
        # print(model.model)
        # print(f'----------------------------------------------------------------------------------------------------')
        # #################################################################################################################
        
        # Split the DataFrame into training and testing datasets (80% train, 20% test)      
        data_train, data_test = train_test_split( self.data, test_size=0.05, random_state=42 ) #, shuffle=False)  
        
        data_train = data_train[self.train_columns_list].reset_index(drop=True).copy()
        data_test  = data_test[self.train_columns_list].reset_index(drop=True).copy()
        
        if self.RESUME_FROM_CHECKPOINT == True:
            print('RESUMED FROM CHECKPOINT')
            model.fit(data=data_train, test_data=data_test, conditional_col='DOY', resume_from_checkpoint=True)
        else:
            model.fit(data=data_train, test_data=data_test, conditional_col='DOY')
        
        # model.save(f'{experiment_dir}/model')
        
        TRAINER_DICT.to_csv(f'{experiment_dir}.csv')        

        if self.report_to == 'wandb':
            # Mark the run as finished
            wandb.finish(exit_code=99, quiet=True)        
        
        ################################################################################################################

class RunTask:
    @staticmethod
    def train_GPT(
                 exp_name = f'A0',                
                 save_steps = 250000,
                 logging_steps = 1000, 
                 TRAINER_RUN = 0,
                 batch_size_list   = [128, 128, 8, 8],
                 max_steps_list    = [300000, 600000, 3000000, 400000, 400000],
                 warmup_steps_list = [200000, 300000, 1000000, 200000, 200000],
                 lr_scheduler_type_list = ['cosine', 'cosine', 'cosine', 'cosine'],
                 num_cycles_list = [1, 1, 2, 5],
                 learning_rate_list = [1e-8, 5e-8, 5e-7, 2e-6, 2e-6],   
                 step_checkpoint_list = [250000, 3000000, 1500000, 400000],
                 resume_from_checkpoint = False,
                 tokenize_it = False,                    
                 report_to = 'none',                      
                 ):
        
        model = ModelBase(
                         EXP_NAME = exp_name,               
                         save_steps = save_steps,
                         logging_steps = logging_steps, 
                         TRAINER_RUN = TRAINER_RUN,                             
                         batch_size_list = batch_size_list,
                         max_steps_list = max_steps_list,
                         warmup_steps_list = warmup_steps_list,
                         lr_scheduler_type_list = lr_scheduler_type_list,
                         num_cycles_list = num_cycles_list,
                         learning_rate_list = learning_rate_list,   
                         step_checkpoint_list = step_checkpoint_list,
                         RESUME_FROM_CHECKPOINT = resume_from_checkpoint,
                         report_to = report_to,
                         )

        if tokenize_it == True:
            model.Tokenize()
            sys.exit(0)
        
        model.train()


if __name__ == "__main__":
    
    freeze_support()
    warnings.filterwarnings("ignore")
    
    fire.Fire(RunTask)
    
    # main()