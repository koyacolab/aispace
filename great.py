import os
import warnings
import json
import typing as tp
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig, TrainerCallback

from transformers.integrations import WandbCallback

from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup 
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from great_dataset import GReaTDataset, GReaTDataCollator
from great_start import (
    GReaTStart,
    CategoricalStart,
    ContinuousStart,
    RandomStart,
    _pad_tokens,
)
from great_trainer import GReaTTrainer
from great_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    _convert_tokens_to_text,
    _convert_text_to_tabular_data,
    _partial_df_to_promts,
    bcolors,
)

from sophia import SophiaG 
from optimizer import SophiaSchedule

import evaluate

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    # neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from sklearn.model_selection import train_test_split

from hls_inference import HLSInference

import numpy as np

from PIL import Image as im 

class GReaT:
    """GReaT Class

    The GReaT class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(
        self,
        llm: str,
        tokenizer: str,
        experiment_dir: str = "trainer_great",
        epochs: int = 100,
        max_steps: int = -1,
        batch_size: int = 8,
        efficient_finetuning: str = "",
        optimizer = 'sophia',
        num_cycles=1,
        **train_kwargs,
    ):
        """Initializes GReaT.

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
            experiment_dir:  Directory, where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            efficient_finetuning: Indication of fune-tuning method
            train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
             see here the full list of all possible values
             https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        """
        # device = torch.device("cuda")
        # Load Model and Tokenizer from HuggingFace
        self.efficient_finetuning = efficient_finetuning
        self.llm = llm
        self.tiktok = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tiktok)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)      #, 
                                                          # attn_pdrop=0.3,) 
                                                          # embd_pdrop=0.3, 
                                                          # resid_pdrop=0.3, 
                                                          # summary_first_dropout=0.3) #.to(device)

        self.model.resize_token_embeddings( len(self.tokenizer) )

        # # ##################################################
        # # vocab = self.tokenizer.get_vocab()
        # # self.model.resize_token_embeddings(len(vocab))
        # # ##################################################

        # #### USE CONFIG TO SET DROPOUTS FOR LAYERS #######
        # config = AutoConfig.from_pretrained(self.llm)

        # print('config:')
        # print(config)
        # print('---------------------------------')
        # fn
        # ##################################################


        if self.efficient_finetuning == "lora":
            # Lazy importing
            try:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    prepare_model_for_int8_training,
                    TaskType,
                )
            except ImportError:
                raise ImportError(
                    "This function requires the 'peft' package. Please install it with - pip install peft"
                )

            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,  # only training 0.16% of the parameters of the model
                lora_alpha=32,
                target_modules=[
                    "c_attn", #"c_proj", "c_fc"
                ],  # this is specific for gpt2 model, to be adapted
                fan_in_fan_out=True,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,  # this is specific for gpt2 model, to be adapted
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_int8_training(self.model)
            # add LoRA adaptor
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        self.optimizer = optimizer
        self.lr_scheduler = self.train_hyperparameters['lr_scheduler_type']
        self.num_cycles = num_cycles

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        test_data: tp.Union[pd.DataFrame, np.ndarray],
        # hls_data, 
        # inf_data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
        # inf_data: tp.Union[pd.DataFrame, np.ndarray],
        # optimizer = '',
        # lr_fit = 2e-5, 
        # LR_SCHEDULER_FIT = 'cosine',
    ) -> GReaTTrainer:
        """Fine-tune GReaT using tabular data.

        Args:
            data: Pandas DataFrame or Numpy Array that contains the tabular data
            column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
            DataFrame, the value is ignored
            conditional_col: If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the last column is considered as conditional feature
            resume_from_checkpoint: If True, resumes training from the latest checkpoint in the experiment_dir.
            If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)

        Returns:
            GReaTTrainer used for the fine-tuning process
        """
        ##### TRAIN DATA #######################################################
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer)
        ########################################################################
        ##### TEST DATA #######################################################
        test_df = _array_to_dataframe(test_data, columns=column_names)
        # self._update_column_information(df)
        # self._update_conditional_information(df, conditional_col)
        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        test_great_ds = GReaTDataset.from_pandas(test_df, split="test")
        test_great_ds.set_tokenizer(self.tokenizer)
        ########################################################################
        
        # test_great_ds_list = []
        # ##### TEST DATA #######################################################
        # for test_data in test_data_list:
        #     test_df = _array_to_dataframe(test_data, columns=column_names)
        #     # self._update_column_information(df)
        #     # self._update_conditional_information(df, conditional_col)
        #     # Convert DataFrame into HuggingFace dataset object
        #     logging.info("Convert data into HuggingFace dataset object...")
        #     test_great_ds = GReaTDataset.from_pandas(test_df, split="test")
        #     test_great_ds.set_tokenizer(self.tokenizer)

        #     test_great_ds_list.append(test_great_ds)
        # ########################################################################

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            max_steps = self.max_steps,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )

        # optimizer = 'sophia'
        if (self.optimizer == 'Sophia') | (self.optimizer == 'sophia'):
            print(f'Optimiser: 2nd order - Sophia')
            print('self.train_hyperparameters:', self.train_hyperparameters)
        ######### Sophia Scheduler #################################
            if len(data) // self.batch_size < 1:
                print('len(data) // self.batch_size < 1')
                fn
            #### CALCULATE total_train_steps #########################################
            total_train_steps = 0
            if self.max_steps < 0:
                total_train_steps = ((len(data) // self.batch_size) + 1) * self.epochs
            else:
                total_train_steps = self.max_steps
                epochs = total_train_steps // ((len(data) // self.batch_size) + 1)
                
            print(f'total_train_steps calculated: step={total_train_steps}, epochs={epochs}, len(data)={len(data)}, batch_size={self.batch_size}')
            print('warmup_steps:', self.train_hyperparameters['warmup_steps'])

            #### SET 2nd ORDER OPTIMIZER ##############################
            self.optimizer = SophiaG(self.model.parameters(), 
                                    lr=self.train_hyperparameters['learning_rate'], 
                                    betas=(0.965, 0.99), 
                                    rho = 0.05, 
                                    weight_decay=2e-1)      

            #### SET lr_scheduler_type ################################
            if self.train_hyperparameters['lr_scheduler_type'] == 'cosine':
                print('lr_scheduler_type: cosine')
                self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, 
                                                               num_warmup_steps=self.train_hyperparameters['warmup_steps'], 
                                                               num_training_steps=total_train_steps,
                                                               num_cycles=self.num_cycles)
            elif self.train_hyperparameters['lr_scheduler_type'] == 'linear':
                print('lr_scheduler_type: linear')
                self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                               num_warmup_steps=self.train_hyperparameters['warmup_steps'], 
                                                               num_training_steps=total_train_steps)
            elif self.train_hyperparameters['lr_scheduler_type'] == 'constant':
                print('lr_scheduler_type: constant')
                self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, 
                                                                 num_warmup_steps=self.train_hyperparameters['warmup_steps'])
            ################# POLYNOMIAL #######################################################################
            elif self.train_hyperparameters['lr_scheduler_type'] == 'polynomial':
                scheduler_type = self.train_hyperparameters['lr_scheduler_type']
                print(f"lr_scheduler_type: {scheduler_type}")
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, 
                                                                         num_warmup_steps=self.train_hyperparameters['warmup_steps'],
                                                                         # power=-1,
                                                                         num_training_steps=total_train_steps,
                                                                         lr_end=self.train_hyperparameters['learning_rate'] / 2) # - \
                                                                            # 0.00001*self.train_hyperparameters['learning_rate'])
            # #######################################################################################################
            # elif self.train_hyperparameters['lr_scheduler_type'] == 'polynomial':
            #     scheduler_type = self.train_hyperparameters['lr_scheduler_type']
            #     print(f"lr_scheduler_type: {scheduler_type}")
            #     self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, 
            #                                                              num_warmup_steps=self.train_hyperparameters['warmup_steps'],
            #                                                              power=-1,
            #                                                              num_training_steps=total_train_steps,
            #                                                              lr_end=1e-9)
            # #################################################################################################

        ############################################################
            
            # fn

            # Setup evaluation 
            # metric = evaluate.load("accuracy")
            # from evaluate.metrics import accuracy
            # metric_mae = evaluate.load("./aispace/metrics/mae")
            # metric_accuracy = evaluate.load("./aispace/metrics/accuracy")

            # def compute_metrics(eval_prediction: EvalPrediction):
            #     # metric = evaluate.load("accuracy")
            #     # print('eval_pred:', p)
            #     logits, labels = eval_prediction
            #     # print('eval_logits:', len(logits), logits)
            #     # print('eval_labels:', len(labels), labels)
            #     predictions = np.argmax(logits, axis=-1)
            #     # print('eval_predictions:', len(predictions), predictions)
            #     # metrics = metric.compute(predictions=predictions[0], references=labels[0])
            #     # print('eval_metrics:', len(metrics), metrics)

            #     # print(self.tokenizer.decode(predictions[0]))
            #     # print(self.tokenizer.decode(labels[0]))
            #     # print('...........................................................................................')
                
            #     # print(self.tokenizer.convert_ids_to_tokens(predictions[0]))
            #     # print(self.tokenizer.convert_ids_to_tokens(labels[0]))
            #     # fn
            #     # predictions = np.argmax(predictions, axis=1)
            #     # metric = evaluate.combine(["accuracy", "mae"])
                                         
            #     return metric_accuracy.compute(predictions=predictions.flatten(), references=labels.flatten())

        ##########################################################################

            from sklearn.metrics import precision_recall_fscore_support, accuracy_score

            def compute_metrics(pred):
                labels = pred.label_ids
                preds = pred.predictions.argmax(-1)
                # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
                acc = accuracy_score(labels, preds)
                return {
                    'accuracy': acc,
                    # 'f1': f1,
                    # 'precision': precision,
                    # 'recall': recall
                }

            class MyCallback(TrainerCallback):
                "A callback that prints a message at the beginning of training"
                def __init__(self, func, imp_data):
                    self.func = func
                    self.sample_dataset = imp_data
                    super().__init__()
            
                def on_step_end(self, args, state, control, **kwargs):
                    ep_list = [x for x in range(10000, state.max_steps, 10000)]
                    if int(state.global_step) in ep_list and state.epoch > 0:
    
                        print(state.epoch)
                        
                        _, to_impute_test = train_test_split( self.sample_dataset, test_size=0.1, random_state=42 )
    
                        to_impute_test  = to_impute_test[['PID', 'DOY', 'B02', 'B03', 'B04']].reset_index(drop=True).copy()
    
                        to_impute = to_impute_test.copy()
                        
                        to_impute[['B02', 'B03', 'B04']] = np.nan
                        # display(to_impute)
                        
                        imputed_data = self.func(df_miss=to_impute, k=1000, max_length=42, temperature=1e-32, device='cuda')
                        # display(imputed_data)
    
                        # Extract the values from the specified columns for both dataframes
                        cols = ['B02', 'B03', 'B04']
                        arr1 = to_impute_test[cols].values
                        arr2 = imputed_data[cols].values
    
                        # Calculate L1 and L2 norms for each row
                        l1_norms = np.linalg.norm(arr1 - arr2, ord=1, axis=1)
                        l2_norms = np.linalg.norm(arr1 - arr2, ord=2, axis=1)
    
                        L1 = np.sum(l1_norms)/len(l1_norms)
                        L2 = np.sum(l2_norms)/len(l2_norms)
    
                        print('L1:', np.sum(l1_norms)/len(l1_norms), ', L2:', np.sum(l2_norms)/len(l2_norms))
                    
                    # fn
            
            great_trainer = GReaTTrainer(
                self.model,
                training_args,
                train_dataset=great_ds,
                eval_dataset={'validation' : test_great_ds}, 
                tokenizer=self.tokenizer,
                data_collator=GReaTDataCollator(self.tokenizer),
                optimizers = (self.optimizer, self.lr_scheduler),
                # callbacks=[MyCallback(func=self.impute, imp_data=test_data)],
                # compute_metrics = compute_metrics,
                )
        else:
                print(f'Optimizer: 1st order')
                # print(f'training_args: {training_args}')
            
                great_trainer = GReaTTrainer(
                self.model,
                training_args,
                train_dataset=great_ds,
                eval_dataset={'validation' : test_great_ds}, 
                tokenizer=self.tokenizer,
                data_collator=GReaTDataCollator(self.tokenizer),
                )

        ###### ADD WANDB CALLBACK ###########################################
        class WandbPredictionProgressCallback(WandbCallback):
            """Custom WandbCallback to log model predictions during training.
        
            This callback logs model predictions and labels to a wandb.Table at each logging step during training.
            It allows to visualize the model predictions as the training progresses.
        
            Attributes:
                trainer (Trainer): The Hugging Face Trainer instance.
                tokenizer (AutoTokenizer): The tokenizer associated with the model.
                sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
                num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
                freq (int, optional): Frequency of logging. Defaults to 2.
            """
        
            def __init__(self, func, trainer, tokenizer, val_dataset, num_samples=100, freq=20):
                """Initializes the WandbPredictionProgressCallback instance.
        
                Args:
                    trainer (Trainer): The Hugging Face Trainer instance.
                    tokenizer (AutoTokenizer): The tokenizer associated with the model.
                    val_dataset (Dataset): The validation dataset.
                    num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
                    freq (int, optional): Frequency of logging. Defaults to 2.
                """
                self.func = func
                super().__init__()
                self.trainer = trainer
                self.tokenizer = tokenizer
                self.sample_dataset = val_dataset
                self.freq = freq

            def on_evaluate(self, args, state, control, **kwargs):
                super().on_evaluate(args, state, control, **kwargs)
                # control the frequency of logging by logging the predictions every `freq` epochs
                # if state.epoch % self.freq == 0:
                    # generate predictions
                    # predictions = self.trainer.predict(self.sample_dataset)
                    # # decode predictions and labels
                    # predictions = decode_predictions(self.tokenizer, predictions)
                    # # add predictions to a wandb.Table
                    # predictions_df = pd.DataFrame(predictions)
                    # predictions_df["epoch"] = state.epoch
                    # records_table = self._wandb.Table(dataframe=predictions_df)
                    # log the table to wandb
                ###### TEST METRICS FOR RANDOM TEST DATASET #############################################################
                ep_list = [x for x in range(10000, state.max_steps, 10000)]
                if int(state.global_step) in ep_list and state.epoch > 0:
                    # to_impute = self.sample_dataset #test_data.copy()

                    _, to_impute_test = train_test_split( self.sample_dataset, test_size=0.2 )

                    to_impute_test  = to_impute_test[['PID', 'DOY', 'B02', 'B03', 'B04']].reset_index(drop=True).copy()

                    to_impute = to_impute_test.copy()
                    
                    to_impute[['B02', 'B03', 'B04']] = np.nan
                    # display(to_impute)
                    
                    imputed_data = self.func(df_miss=to_impute, k=1000, max_length=33, temperature=1e-32, device='cuda')
                    # display(imputed_data)

                    #### TEST NaNs: Set B02 as NaN in a random row
                    # random_row_index = np.random.choice(imputed_data.index)
                    # imputed_data.at[random_row_index, 'B02'] = np.nan

                    # imputed_data = imputed_data.dropna()

                    if len(imputed_data) != len(to_impute_test):
                        print('imputed_data != nan_data_doy: ', len(to_impute_test) - len(imputed_data))
                        # Use the merge function with indicator=True
                        original_df = to_impute_test
                        subset_df = imputed_data
            
                        merged_df = pd.merge(original_df, subset_df, on=['PID', 'DOY'], how='left', indicator=True)
            
                        # Find the rows in original_df that are not in subset_df
                        missing_rows = original_df[merged_df['_merge'] == 'left_only']
            
                        # Display the missing rows
                        # display(missing_rows)
            
                        imputed_data = pd.concat([imputed_data, missing_rows], axis=0)

                    # Extract the values from the specified columns for both dataframes
                    cols = ['B02', 'B03', 'B04']
                    arr1 = to_impute_test[cols].values
                    arr2 = imputed_data[cols].values

                    # Calculate L1 and L2 norms for each row
                    l1_norms = np.linalg.norm(arr1 - arr2, ord=1, axis=1)
                    l2_norms = np.linalg.norm(arr1 - arr2, ord=2, axis=1)

                    L1 = np.sum(l1_norms)/len(l1_norms)
                    L2 = np.sum(l2_norms)/len(l2_norms)

                    L1 = np.nan_to_num(L1, nan=0)
                    L2 = np.nan_to_num(L2, nan=0)

                    print('L1:', L1, ', L2:', L2)
                    
                    self._wandb.log({"L1": L1})
                    self._wandb.log({"L2": L2})
                ##########################################################################################
                # ###### TEST METRICS FOR INFERENCE TEST DATASET #############################################################
                # ep_list_2 = [x for x in range(1000, state.max_steps, 100000)]
                # # print(ep_list_2)
                # if int(state.global_step) in ep_list_2 and state.epoch > 0:
                #     hls_data = HLSInference(table_dtype = int)
                #     hls_data.clip_dataset(x1=50.0, y1=50.0, x2=100.0, y2=100.0)
                #     doys = [211,]
                #     _ = hls_data._get_data_doys(doys = doys, SHOW = False)
                #     _, _ = hls_data._nan_9999()
                #     _, _ = hls_data._set_clear_cloud()
                #     _, _, _, _ = hls_data._set_train_columns_name(SHOW = False)
                #     _, _ = hls_data._set_train_test_data(doy=211.0, x1=45.0, y1=45.0, x2=50.0, y2=50.0, for_show_nan=False)
                #     _ = hls_data._to_impute(SHOW = False)
                #     hls_data._inference_train_test_data()
                #     train_columns_list = ['PID', 'DOY', 'B02', 'B03', 'B04', ]
                #     recovered_data = hls_data._impute(model=self.func, columns_impute=train_columns_list, k=10000, max_length=36, temperature=1e-32)
                #     hls_data._set_inference_recovered()
                #     img_list = hls_data._inference_imshow()

                #     # print('img_list:', type(img_list[0]), img_list[0].shape)
                    
                #     images = self._wandb.Image(img_list[0], caption="original")
                #     print('images:', type(images), type(img_list[1]), img_list[1].shape)

                #     nr, nc,  _= img_list[1].shape
                #     shrinkFactor = 5
                #     img_pil = im.fromarray(img_list[1]) 
                #     img_pil = img_pil.resize((round(nc*shrinkFactor),round(nr*shrinkFactor)))
                #     # img_resized = np.array(img_pil)
                #     # saving the final output  
                #     # as a PNG file 
                #     img_pil.save(f'{int(state.global_step)}_pic.png') 
                #     # self._wandb.log({"original": images})
                #     # images = self._wandb.Image(img_list[1], caption="imputed")
                #     # self._wandb.log({"imputed": images})
                #     fn
                # #     # print()
        
        #####################################################################
        # Instantiate the WandbPredictionProgressCallback
        progress_callback = WandbPredictionProgressCallback(
            trainer=great_trainer,
            tokenizer=self.tokenizer,
            val_dataset=test_data,
            func = self.impute,
            # num_samples=10,
            # freq=20,
        )
        
        # # Add the callback to the trainer
        great_trainer.add_callback(progress_callback)
        ####################################################################

        #### START TRAINING ################################################
        # Start training
        logging.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def great_sample(
        self,
        starting_prompts: tp.Union[str, list[str]],
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """Generate synthetic tabular data samples conditioned on a given input.

        Args:
            starting_prompts: String or List of Strings on which the output is conditioned.
             For example, "Sex is female, Age is 26"
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process.
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.

         Returns:
            Pandas DataFrame with synthetic data generated based on starting_prompts
        """
        # ToDo: Add n_samples argument to generate more samples for one conditional input.

        from transformers import set_seed
        set_seed(42)

        self.model.to(device)
        starting_prompts = (
            [starting_prompts]
            if isinstance(starting_prompts, str)
            else starting_prompts
        )
        generated_data = []

        # Generate a sample for each starting point
        if len(starting_prompts) > 1:
            loop_iter = tqdm(starting_prompts)
        else:
            loop_iter = starting_prompts
        for prompt in loop_iter:
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)

            # # ################################################################################
            # print('loop_iter:', loop_iter)
            # print('prompt:', prompt)
            # # print('start_token:', start_token)
            # print('tokenizer.tokenize:', self.tokenizer.tokenize(prompt, padding=True))
            # # # fn 
            # # ###########################################################################
            
            # Generate tokens
            gen = self.model.generate(
                input_ids=torch.unsqueeze(start_token, 0),
                max_length=max_length,
                do_sample=True,
                # top_k=0,  
                temperature=temperature,
                pad_token_id=50256,
            )
    
            generated_data.append(torch.squeeze(gen))

            # # ##########################################################################
            # # print('generated_data:', gen)
            # # text_data = [self.tokenizer.decode(t) for t in generated_data]
            # tokens = [self.tokenizer.convert_ids_to_tokens(t) for t in generated_data]
            # print('gen tokens:', len(tokens[0]), tokens)

            # # Convert tokens to text
            # text_data = [self.tokenizer.decode(t) for t in gen]
        
            # # Clean text
            # text_data = [d.replace("<|endoftext|>", "") for d in text_data]
            # text_data = [d.replace("\n", " ") for d in text_data]
            # text_data = [d.replace("\r", "") for d in text_data]
            # print('text_data:', text_data)
            # # fn
            # # ##########################################################################

        # Convert Text back to Tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        df_gen = _convert_text_to_tabular_data(decoded_data, self.columns)

        # # ###############################################################################
        # print('decoded_data:', decoded_data)
        # print('------------------------------------')
        # fn
        # # ###########################################################################

        return df_gen

    def impute(
        self,
        df_miss: pd.DataFrame,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        max_retries=15,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """Impute a DataFrame with missing values using a trained GReaT model.
        Args:
            df_miss: pandas data frame of the exact same format (column names, value ranges/types) as the data that
             was used to train the GReaT model, however some values might be missing, which is indicated by the value of NaN.
             This function will sample the missing values conditioned on the remaining values.
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process
            k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information!
            device: Set to "cpu" if the GPU should not be used. You can also specify the specific GPU to run on.

        Returns:
            Pandas DataFrame with n_samples rows of generated data
        """

        # print('impute')
        # fn
        
        # Check DataFrame passed.
        if set(df_miss.columns) != set(self.columns):
            raise ValueError(
                "The column names in the DataFrame passed to impute do not match the columns of the GReaT model."
            )

        self.model.to(device)

        # start_token = torch.tensor(_pad_tokens(self.tokenizer(starting_prompts)["input_ids"])).to(device)
        index = 0
        df_list = []
        with tqdm(total=len(df_miss)) as pbar:
            while index < len(df_miss):
                is_complete = False
                retries = 0
                df_curr = df_miss.iloc[[index]]
                org_index = df_curr.index  # Keep index in new DataFrame
                while not is_complete:
                    num_attrs_missing = pd.isna(df_curr).sum().sum()
                    # Generate text promt from current features.
                    starting_prompts = _partial_df_to_promts(df_curr)

                    # # ##########################################################
                    # print("Number of missing values: ",  num_attrs_missing)
                    # display('df_miss:', df_miss)
                    # display('df_curr:', df_curr)
                    # print('starting_prompts:', starting_prompts)
                    # # ##########################################################
                    
                    df_curr = self.great_sample(
                        starting_prompts, temperature, max_length, device=device
                    )

                    # # ##########################################################
                    # print('predict:') 
                    # display(df_curr)
                    # # fn
                    # # ##########################################################

                    # Convert numerical values to float, flawed numerical values to NaN
                    for i_num_cols in self.num_cols:
                        df_curr[i_num_cols] = pd.to_numeric(
                            df_curr[i_num_cols], errors="coerce"
                        )
                    df_curr[self.num_cols] = df_curr[self.num_cols].astype(np.float)
                    # df_curr[self.num_cols] = df_curr[self.num_cols].astype(int)

                    # Check for missing values
                    nans = df_curr.isna()
                    if not df_curr.isna().any().any():
                        is_complete = True
                        df_list.append(df_curr.set_index(org_index))
                    else:
                        retries += 1
                        ############
                        print(f'retries starting_prompts: {retries} : {starting_prompts}')
                        display('retries predict:', df_curr)
                        ############
                    if retries == max_retries:
                        ############
                        print('Max retries starting_prompts:', starting_prompts)
                        display('predict:', df_curr)
                        ############
                        warnings.warn(" reached.")
                        break
                index += 1
                pbar.update(1)

        # fn
        return pd.concat(df_list, axis=0)

    def save(self, path: str):
        """Save GReaT Model

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        # Make directory
        if os.path.isdir(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            # NDArray is not JSON serializable and therefore has to be converted into a list.
            if isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(
                    attributes["conditional_col_dist"]
                )

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_finetuned_model(self, path: str):
        """Load fine-tuned model

        Load the weights of a fine-tuned large language model into the GReaT pipeline

        Args:
            path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        """Load GReaT class

        Load trained GReaT model from directory.

        Args:
            path: Directory where GReaT model is saved

        Returns:
            New instance of GReaT loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_great model instance
        great = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        return great

    def _update_column_information(self, df: pd.DataFrame):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(
        self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None
    ):
        assert conditional_col is None or isinstance(
            conditional_col, str
        ), f"The column name has to be a string and not {type(conditional_col)}"
        assert (
            conditional_col is None or conditional_col in df.columns
        ), f"The column name {conditional_col} is not in the feature names of the given dataset"

        # Take the distribution of the conditional column for a starting point in the generation process
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(
        self,
        start_col: tp.Optional[str],
        start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]],
    ) -> GReaTStart:
        if start_col and start_col_dist is None:
            raise ValueError(
                f"Start column {start_col} was given, but no corresponding distribution."
            )
        if start_col_dist is not None and not start_col:
            raise ValueError(
                f"Start column distribution {start_col} was given, the column name is missing."
            )

        assert start_col is None or isinstance(
            start_col, str
        ), f"The column name has to be a string and not {type(start_col)}"
        assert (
            start_col_dist is None
            or isinstance(start_col_dist, dict)
            or isinstance(start_col_dist, list)
        ), f"The distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}"

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)


###########################################################################################

    def sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """
        Generate synthetic tabular data samples.

        Args:
            n_samples (int): Number of synthetic samples to generate.
            start_col (str, optional): Feature to use as the starting point for the generation process.
                Defaults to the target learned during fitting if not provided.
            start_col_dist (dict or list, optional): Feature distribution of the starting feature.
                For discrete columns, should be in the format "{F1: p1, F2: p2, ...}".
                For continuous columns, should be a list of possible values.
                Defaults to the target distribution learned during fitting if not provided.
            temperature (float): Controls the softmax function for token sampling.
                Lower values make it sharper (0 equals greedy search), higher values introduce more diversity but also uncertainty.
            k (int): Sampling batch size. Higher values speed up the generation process.
            max_length (int): Maximum number of tokens to generate. Ensure it's long enough to not cut off any information.
            drop_nan (bool): Whether to drop rows with NaN values. Defaults to False.
            device (str): Device to use for generation. Set to "cpu" to avoid using GPU. Specific GPU can also be named.

        Returns:
            pd.DataFrame: DataFrame containing n_samples rows of generated data.
        """
        great_start = self._get_start_sampler(start_col, start_col_dist)

        # Move model to device
        self.model.to(device)

        # Init list for generated DataFrames
        dfs = []

        # Start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            _cnt = 0
            try:
                while n_samples > already_generated:
                    start_tokens = great_start.get_start_tokens(k)
                    start_tokens = torch.tensor(start_tokens).to(device)

                    # Generate tokens
                    tokens = self.model.generate(
                        input_ids=start_tokens,
                        max_length=max_length,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=50256,
                    )

                    # Convert tokens back to tabular data
                    text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                    df_gen = _convert_text_to_tabular_data(text_data, self.columns)

                    # Remove rows where we have not generated anything
                    df_gen = df_gen[~(df_gen == "placeholder").any(axis=1)]

                    # Remove rows where all values are NaN
                    df_gen = df_gen.dropna(how="all")

                    # Optional: Remove rows with any NaN values
                    if drop_nan:
                        df_gen = df_gen.dropna()

                    # Remove rows with flawed numerical values but keep NaNs
                    for i_num_cols in self.num_cols:
                        coerced_series = pd.to_numeric(
                            df_gen[i_num_cols], errors="coerce"
                        )
                        df_gen = df_gen[
                            coerced_series.notnull() | df_gen[i_num_cols].isna()
                        ]

                    # Convert numerical columns to float
                    df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                    dfs.append(df_gen)
                    already_generated += len(dfs[-1])

                    # Update progress bar
                    pbar.update(len(dfs[-1]))

                    # Check if we are actually generating synthetic samples and if not, break everything
                    _cnt += 1
                    if _cnt > 13 and already_generated == 0:
                        raise Exception("Breaking the generation loop!")

            except Exception as e:
                print(f"{bcolors.FAIL}An error has occurred: {str(e)}{bcolors.ENDC}")
                print(
                    f"{bcolors.WARNING}To address this issue, consider fine-tuning the GReaT model for an longer period. This can be achieved by increasing the number of epochs.{bcolors.ENDC}"
                )
                print(
                    f"{bcolors.WARNING}Alternatively, you might consider increasing the max_length parameter within the sample function. For example: model.sample(n_samples=10, max_length=2000){bcolors.ENDC}"
                )
                print(
                    f"{bcolors.OKBLUE}If the problem persists despite these adjustments, feel free to raise an issue on our GitHub page at: https://github.com/kathrinse/be_great/issues{bcolors.ENDC}"
                )

        df_gen = pd.concat(dfs)
        df_gen = df_gen.reset_index(drop=True)
        return df_gen.head(n_samples)
