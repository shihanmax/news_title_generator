import logging
import re

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config
from .utils import Vocab

logger = logging.getLogger(__name__)


class TextPair(object):
    """Maintains a text pair: (raw_text, simplified text)."""

    def __init__(self, source, simplified, tokenize):
        """Tokenize source and simplified text into list of strings."""
        self.source = tokenize(source)
        self.simplified = tokenize(simplified)
        self.src = source
        self.tgt = simplified


class RawDataProvider(object):
    """加载原始训练数据，并分割为训练、测试、验证集."""

    def __init__(self, config: Config):
        self.config = config

    def load_raw_data(self):
        

        return train_set, valid_set, test_set

    def build_data_to_text_pairs(self, questions):
        text_pairs = []
        # step_simp_pairs = []

        for question in tqdm(questions):
            thought_info = (
                ThoughtHandler.collect_thought_info_of_question(question)
            )

            thoughts = thought_info["thoughts"]

            for thought in thoughts:

                short_description = thought["short_description"]
                guide = thought["guide"]
                question = thought["question"]
                step_text = thought["step_text"]

                if step_text is None or step_text.count("$") % 2 != 0:
                    continue

                if not step_text.strip() or not short_description.strip():
                    continue

                # step_simp_pairs.append(
                #     TextPair(step_text, short_description),
                # )

                # continue

                if question:
                    guide = question

                if not guide:
                    continue

                # 删除文本中的空格、换行、tab...
                guide = re.sub(r"[\n\t ]", "", guide)
                short_description = re.sub(r"[\n\t ]", "", short_description)
                new_pair = TextPair(guide, short_description)

                if not new_pair.simplified or not new_pair.source:
                    continue
                text_pairs.append(new_pair)

        return text_pairs


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
        src_tokenized = text_pair.source[: self.config.max_src_len]
        tgt_tokenized = text_pair.simplified[: self.config.max_tgt_len - 1]

        src_token_ids = []
        src_token_ids_with_oov = []

        for token in src_tokenized:
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

        for token in tgt_tokenized:
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
