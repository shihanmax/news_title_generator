import logging
import re
import json
import torch
import random
from collections import Counter
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config
from .utils import Vocab, build_vocab

logger = logging.getLogger(__name__)


class TextPair(object):
    """Maintains a text pair: (raw_text, simplified text)."""

    def __init__(self, source, target, tokenize):
        """Tokenize source and simplified text into list of strings."""
        self.tokenized_source = tokenize(source)
        self.tokenized_target = tokenize(target)
        self.source = source
        self.target = target


class RawDataProvider(object):
    """加载原始训练数据，并分割为训练、测试、验证集."""

    def __init__(self, config: Config):
        self.config = config

    def _convert_news_to_pairs(self, news, tokenizer, create_vocab=False):
        logger.info("start loading raw data...")
        max_src_len = -1
        max_tgt_len = -1
        
        frequency = Counter()
        text_pairs = []
        
        for n in tqdm(news):
            title = n["title"].strip()
            content = n["content"].strip()

            if not title or not content:
                continue
            
            title = re.sub(r"[\n\t ]", "", title)
            content = re.sub(r"[\n\t ]", "", content)
            
            if create_vocab:
                frequency.update(Counter(tokenizer(title)))
                frequency.update(Counter(tokenizer(content)))
            
            text_pairs.append(
                TextPair(source=content, target=title, tokenize=tokenizer)
            )
            
            max_src_len = max(max_src_len, len(tokenizer(content)))
            max_tgt_len = max(max_tgt_len, len(tokenizer(title)))
        
        if create_vocab:
            vocab = build_vocab(
                raw_text_list=None, frequency=frequency, 
                threshold=self.config.vocab_freq, tokenizer=tokenizer,
            )
            self.config.vocab_size = vocab.size
            logger.info(f"config: vocab_size set to :{vocab.size}")
        else:
            vocab = None
        
        logger.info(f"max_src_len:{max_src_len}, max_tgt_len:{max_tgt_len}")
        
        return text_pairs, vocab
                
    def load_raw_data(self, tokenizer, create_vocab):
        with open(self.config.raw_data_path) as frd:
            all_news = json.load(frd)
        
        random.seed(self.config.random_seed)
        random.shuffle(all_news)
        
        if self.config.random_sample_size:
            logger.info(f"采样{self.config.random_sample_size}条数据")
            all_news = all_news[: self.config.random_sample_size]
            
        all_text_pairs, vocab = self._convert_news_to_pairs(
            all_news, tokenizer, create_vocab,
        )
        
        total = len(all_text_pairs)

        logger.info(f"total samples:{total}")
        len_test = round(self.config.test_ratio * total)
        len_valid = round(self.config.valid_ratio * total)
        len_train = total - len_test - len_valid
        
        train_set = all_text_pairs[:len_train]
        valid_set = all_text_pairs[len_train: len_train + len_valid]
        test_set = all_text_pairs[-len_test:]
        
        logger.info(
            f"Done splitting data: train:{len(train_set)}, "
            f"valid:{len(valid_set)}, test:{len(test_set)}"
        )

        return train_set, valid_set, test_set, vocab
    

class PtrDataset(Dataset):

    def __init__(self, config, vocab: Vocab, text_pairs):
        self.config = config
        self.vocab = vocab

        self.text_pairs = text_pairs

        self.oov_mapper = {}

    def fetch_and_clear_mapper(self):
        """Get a copy of the out-of-vocab mapper, then clear the original one.

        This method should be called every batch, to make sure that oov words
        from the batch shares the same mapper ids.

        Returns:
            dict: oov mapper: {"oov_token": id_, ...}
        """
        mapper = {k: v for k, v in self.oov_mapper.items()}
        self.oov_mapper.clear()

        return mapper

    def get_max_oov_count_curr_batch(self):
        return len(self.oov_mapper)

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        return self.build_single_text_pair(self.text_pairs[idx])

    def build_single_text_pair(self, text_pair: TextPair):

        # trunc first
        src = text_pair.tokenized_source[: self.config.max_src_len]
        tgt = text_pair.tokenized_target[: self.config.max_tgt_len - 1]

        src_token_ids = []
        src_token_ids_with_oov = []

        for token in src:
            if token not in self.vocab.str2idx:
                src_token_ids.append(self.vocab.unk_idx)

                if token not in self.oov_mapper:
                    oov_id = self.vocab.size + len(self.oov_mapper)
                    self.oov_mapper[token] = oov_id
                else:
                    oov_id = self.oov_mapper[token]

                src_token_ids_with_oov.append(oov_id)

            else:
                src_token_ids.append(self.vocab.str2idx[token])
                src_token_ids_with_oov.append(self.vocab.str2idx[token])

        tgt_token_ids = []
        tgt_token_ids_with_oov = []

        for token in tgt:
            if token not in self.vocab.str2idx:
                tgt_token_ids.append(self.vocab.unk_idx)

                if token in self.oov_mapper:
                    oov_id = self.oov_mapper[token]
                else:
                    oov_id = self.vocab.unk_idx  # 摘要中的OOV（除原文中的）

                tgt_token_ids_with_oov.append(oov_id)
            else:
                tgt_token_ids.append(self.vocab.str2idx[token])
                tgt_token_ids_with_oov.append(
                    self.vocab.str2idx[token],
                )

        # add <S> and <E> for decoding
        # tgt_token_ids.insert(0, self.vocab.sos_idx)
        # tgt_token_ids_with_oov.insert(0, self.vocab.sos_idx)

        if len(tgt_token_ids) < self.config.max_tgt_len:
            tgt_token_ids.append(self.vocab.eos_idx)
            tgt_token_ids_with_oov.append(self.vocab.eos_idx)

        src_valid_len = len(src_token_ids)
        tgt_valid_len = len(tgt_token_ids)

        # padding
        src_token_ids.extend(
            [self.vocab.pad_idx] * (self.config.max_src_len - src_valid_len),
        )

        src_token_ids_with_oov.extend(
            [self.vocab.pad_idx] * (self.config.max_src_len - src_valid_len),
        )

        tgt_token_ids.extend(
            [self.vocab.pad_idx] * (self.config.max_tgt_len - tgt_valid_len),
        )

        tgt_token_ids_with_oov.extend(
            [self.vocab.pad_idx] * (self.config.max_tgt_len - tgt_valid_len),
        )

        data = {
            "src_token_ids": src_token_ids,
            "src_token_ids_with_oov": src_token_ids_with_oov,
            "tgt_token_ids": tgt_token_ids,
            "tgt_token_ids_with_oov": tgt_token_ids_with_oov,
            "src_valid_len": torch.tensor(src_valid_len, dtype=torch.int64),
            "tgt_valid_len": torch.tensor(tgt_valid_len, dtype=torch.int64),
        }

        data = {
            k: torch.tensor(v, dtype=torch.long)
            if not isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        return data
