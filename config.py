import os
from enum import Enum

import torch


class Phase(Enum):
    """Mark training phase."""

    TRAIN = 1
    VALID = 2
    TEST = 3


class Config(object):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "./resource/data/news.xml")
    model_path = os.path.join(base_dir, "./output/model")
    inference_model_path = os.path.join(base_dir, "./output/model.ep")
    summary_writer_path = os.path.join(base_dir, "./output/summary")

    max_news_len = 2000  # 新闻最大长度 TODO
    max_title_len = 20  # 最大新闻标题长度，包含<S>, <E>
    max_oov_num = 30  # 新闻中最多存在的未登陆词的数量

    random_sample_size = None
    test_ratio = 0.1
    valid_ratio = 0.1

    # lstm config
    vocab_size = 1814  # TODO 
    embedding_dim = 256
    hidden = 256

    # decoder settings
    decoder_input_dim = 256
    use_ptr = True
    max_decode_len = 20  # 最大解码长度
    beam_size = 3
    decode_top_k = 1
    teacher_forcing = 0.8  # 执行teacherforcing的概率，设置为0则关闭

    # hyper-params of training
    lr = 3e-3
    betas = (0.9, 0.999)
    epoch = 100
    warmup_epochs = 5
    batch_size = 128 if not random_sample_size else 3
    gradient_clip = 10
    not_early_stopping_at_first = 10
    es_with_no_improvement_after = 10

    # msg settings
    verbose = 10
    show_decode = 50

    beam_search_sample_count = 6
