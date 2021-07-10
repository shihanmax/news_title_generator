import logging

from collections import Counter

logger = logging.getLogger(__file__)


class Vocab(object):
    
    def __init__(
        self, str2idx, idx2str, pad_idx, unk_idx, sos_idx, eos_idx, size
    ):
        self.str2idx = str2idx
        self.idx2str = idx2str
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.size = size


def build_vocab(
    raw_text_list, frequency=None, keep_tokens=("PAD", "UNK", "SOS", "EOS"), 
    ignore_tokens=("\n", "\t", "\r", " "), threshold=-1, 
    tokenizer=lambda x: list(x),
):
    r"""Build vocab with given text list.

    Args:
        raw_text_list (List): raw text list
        ignore_tokens (tuple, optional): token to ignore. 
            Defaults to ("\n", "\t", "\r", " ").
        threshold (int, optional): ignore tokens by frequency less than. 
            Defaults to -1, means do not ignore any tokens by frequency.
        tokenizer (Callable, optional): tokenization method. 
            Defaults to token-wise split. 
            (which accept a string and return a list of string.)
        
    Returns:
        str2idx: (Dict), str to idx mapper
        idx2str: (Dict), idx to str mapper
    """
    if not frequency:
        frequency = Counter()
        
        for text in raw_text_list:
            frequency.update(Counter(tokenizer(text)))

    # collect tokens to delete
    token_to_del = []
    for k, v in frequency.items():
        if k in ignore_tokens or (threshold > 0 and v < threshold):
            token_to_del.append(k)
    
    # delete tokens
    for token in token_to_del:
        del frequency[token]

    str2idx = {}
    idx2str = {}
    
    for idx, token in enumerate(keep_tokens):
        str2idx[token] = idx
        idx2str[idx] = token
        
    for idx, (token, _) in enumerate(
        frequency.items(), start=len(keep_tokens)
    ):
        str2idx[token] = idx
        idx2str[idx] = token
    
    logger.info(f"Vocab built: length:{len(str2idx)}, freq limit:{threshold}")
    
    return Vocab(str2idx, idx2str, 0, 1, 2, 3, len(str2idx))


def translate_logits(idx2str, oov_mapper, token_ids, num):
    """Convert logits to text.

    Args:
        idx2str (dict): mapper from token_ids to tokens
        oov_mapper (dict): mapper from token_ids to oov tokens
        token_ids (Tensor): bs, max_len
        num (int): number of instances to return
    """
    max_len = token_ids.shape[0]

    if num == -1:
        num = max_len

    token_ids = token_ids[:num, :]

    res = []
    for i in range(num):
        tokens = []
        for token_id in token_ids[i, :]:
            token_id = int(token_id)

            if token_id in {0, 2, 3}:
                # skip the 'pad', SOS, EOS
                continue

            if token_id in oov_mapper:
                tokens.append(oov_mapper[token_id])
            elif token_id in idx2str:
                tokens.append(idx2str[token_id])
            else:
                tokens.append("ERROR")

        res.append(tokens)

    return res
