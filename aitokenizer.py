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

class TrainTokenizer(TrainCorpus):
    def __init__(self, data, old_tokenizer):
        super().__init__(data)

        self.old_tokenizer = old_tokenizer
        self.tokenizer = None
        
    def _train(self, ):
        self.tokenizer = self.old_tokenizer.train_new_from_iterator(self._get_training_corpus(), 52000)
        return self.tokenizer