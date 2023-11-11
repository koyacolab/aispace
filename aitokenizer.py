from datasets import Dataset
import random

class TrainCorpus:
    def __init__(self, data):
        
        self.great_ds = Dataset.from_pandas(data)
        # print('dataset.shape:', self.great_ds.shape)

        self.texts_list = []
        self.training_corpus = []
        
    def _to_text(self, volume=1):
    
        texts_list = []
        
        num_rows = self.great_ds.num_rows
    
        for jj in range(volume):
    
            for ii in range(num_rows):
                # If int, what else?
                row = self.great_ds._data.fast_slice(ii, 1)
                
                # ####### ORIGINAL SHUFFLING ##############################
                shuffle_idx = list(range(row.num_columns))
                random.shuffle(shuffle_idx)
                
                shuffled_text = ", ".join(
                    [
                        "%s is %s"
                        % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                        for i in shuffle_idx
                    ]
                )
        
                texts_list.append(shuffled_text)

            self.texts_list = texts_list
    
        return texts_list

    def _get_training_corpus(self,):

        texts_list = self._to_text(volume=1)
        
        # print(type(texts_list), len(texts_list))
        
        training_corpus = (
            texts_list[i : i + 1]
            for i in range(0, len(texts_list), 1)
        )

        self.training_corpus = training_corpus

        return training_corpus

from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE

# Add pretokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit, Digits, ByteLevel
from tokenizers.trainers import WordLevelTrainer, BpeTrainer

from transformers import PreTrainedTokenizerFast

from transformers import AutoTokenizer

class TrainTokenizer(TrainCorpus):
    def __init__(self, data, test_sample, old_tokenizer):
        super().__init__(data)

        self.test_sample = test_sample

        self.old_tokenizer = old_tokenizer
        # self.tokenizer = None
        
        # We need to specify the UNK token
        # self.new_tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))
        # self.new_tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))
        self.new_tokenizer = Tokenizer(model=BPE(unk_token="[UNK]"))
        # print('1')
        # self.new_tokenizer.pre_tokenizer = WhitespaceSplit()
        self.new_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        # self.new_tokenizer.pre_tokenizer = Digits(individual_digits=True)
        # Let's test our pre_tokenizer
        print(self.new_tokenizer.pre_tokenizer.pre_tokenize_str(test_sample))
        # fn
        
    def _train(self, special_tokens=["DOY", "PID"]):
        
        # trainer = WordLevelTrainer(
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "is"] + special_tokens
        )
        self.new_tokenizer.train_from_iterator(self._get_training_corpus(), trainer=trainer)
        

        self.new_tokenizer.save("tokenizer.json")
        # print('is_fast:', self.new_tokenizer.is_fast)
        # fn
        self.new_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        self.new_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.new_tokenizer.pad_token = self.new_tokenizer.eos_token
        self.new_tokenizer.save_pretrained("aispace-tokenizer")
        load_tokenizer = AutoTokenizer.from_pretrained("aispace-tokenizer")
        # load_tokenizer(self.test_sample)
        # self.tokenizer = self.old_tokenizer.train_new_from_iterator(self._get_training_corpus(), 52000)
        return load_tokenizer

class PreTrainTokenizer(TrainCorpus):
    def __init__(self, data, old_tokenizer):
        super().__init__(data)

        self.old_tokenizer = old_tokenizer
        self.tokenizer = None
        
    def _train(self, special_tokens=["DOY", "PID"]):
        self.tokenizer = self.old_tokenizer.train_new_from_iterator(self._get_training_corpus(),
                                                                    52000,
                                                                    new_special_tokens=special_tokens) #+['is', ','],)
        
        return self.tokenizer





