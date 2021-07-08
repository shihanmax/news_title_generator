import json
import logging
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append("..")
from thought_short_description_e2e.config import Config
from thought_short_description_e2e.data import PtrSumDataset, TextPair
from thought_short_description_e2e.trainer.trainer import Trainer, Tester
from thought_short_description_e2e.vocab import Vocab

torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO)




# vocab = Vocab(Config)

# train_data_loader = json_to_data_loader(
#     vocab, "./resource/0425_no_option/train_data.json",
# )

# valid_data_loader = json_to_data_loader(
#     vocab, "./resource/0425_no_option/valid_data.json",
# )

# test_data_loader = json_to_data_loader(
#     vocab, "./resource/0425_no_option/test_data.json",
# )

# trainer = Trainer(
#     config=Config,
#     train_data_loader=train_data_loader,
#     valid_data_loader=valid_data_loader,
#     test_data_loader=test_data_loader,
# )

# trainer.start_train()


src = []
tgt = []

with open("./resource/0425_no_option/test_data.json") as frd:
    all_ = json.load(frd)

for i in all_:
    src.append(i["src"])
    tgt.append(i["tgt"])

tester = Tester(Config)

res = tester.predict_to_file(
    text_list=src,
    real_text_list=tgt,
    export_to="0426.txt",
    calc_rouge=True,
)


# result = tester.predict_text_list(
#     [
#         "根据图表给出的数据估计可得:摸到黑球的频率为$0.25$ $ \\therefore$相同条件下,摸到黑球的概率为$0.25$",
#     ],
#     real_text_list=None,
# )

# print(result['gen_text_list'][0])
