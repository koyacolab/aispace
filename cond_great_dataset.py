import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_tokenizer(self, tokenizer):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        # # ####### ORIGINAL SHUFFLING ##############################
        # shuffle_idx = list(range(row.num_columns))
        # random.shuffle(shuffle_idx)
        
        # ################ SET & SHUFFLING SEEN AND UNSEEN VARIABLES ####################### 
        shuffle_idx1 = list(range(row.num_columns))[0:2]
        # random.shuffle(shuffle_idx1)
        
        shuffle_idx2 = list(range(row.num_columns))[2:]
        random.shuffle(shuffle_idx2)
        
        ####### CONVERT SEEN VARIABLE TO SHUFFLED TEXT #################
        shuffled_text1 = ", ".join(
            [
                "%s is %s"
                % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                for i in shuffle_idx1
            ]
        )
        ####### ADD COMMA & SPACE AT THE END OF SEEN DATA ##############
        shuffled_text1 = shuffled_text1 + ", "

        ####### CONVERT UNSEEN VARIABLE TO SHUFFLED TEXT ###############
        shuffled_text2 = ", ".join(
            [
                "%s is %s"
                % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                for i in shuffle_idx2
            ]
        )

        tokenized_text = self.tokenizer(shuffled_text1, shuffled_text2, return_token_type_ids=True, padding=True)
        
        # tokenized_text = self.tokenizer(shuffled_text1, shuffled_text2, padding=True)
        print(tokenized_text)
        fn

        # ##### CHECK INPUTS ################################################################
        # #### check shuffled_text ###########
        # print('\n -------------- DATASET: CHECK TOKENIZER --------------------------------')
        # print(f'\n [{shuffled_text}]')
        # # print(f'\n [{shuffled_text2}]')
        # ###################################
        # # print(key, {type(key)}, {key}')
        # # fn        
        
        # tokenized_text = self.tokenizer.tokenize(shuffled_text, padding=True)
        # print(f'\n {len(tokenized_text)}: [{tokenized_text}]')
        # print('\n ------------------------------------------------------------------------')
        # fn
        # ####################################################################################      
        
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        # print('Collator:', features)
        # fn
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch


######################################################################  

class GReaTestDataset(Dataset):
    """GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_tokenizer(self, tokenizer):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        # ####### ORIGINAL SHUFFLING ##############################
        shuffle_idx = list(range(row.num_columns))
        # random.shuffle(shuffle_idx)
        # ######## SHUFFLING ONLY IMPUTED COLUMNS #########################################
        # shuffle_idx = list(range(row.num_columns))[-4:]
        # random.shuffle(shuffle_idx)
        # shuffle_idx = list(range(row.num_columns))[:-4] + shuffle_idx
        # #################################################################################

        
        shuffled_text = ", ".join(
            [
                "%s is %s"
                % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                for i in shuffle_idx
            ]
        )

        tokenized_text = self.tokenizer(shuffled_text, padding=True)

        ##### CHECK INPUTS ############################################################
        #### check shuffled_text ###########
        print(f'[{shuffled_text}]')
        ###################################
        # print(key, {type(key)}, {key}')
        # fn        
        
        tokenized_text = self.tokenizer.tokenize(shuffled_text, padding=True)
        print(f'[{tokenized_text}]')
        fn
        #################################################################################
        
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)
