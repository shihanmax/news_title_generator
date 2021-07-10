import os
import json
import torch
import logging
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nlkit.utils import (
    weight_init, get_linear_schedule_with_warmup_ep, 
    check_should_do_early_stopping, Phase
)
from nlkit.metrics import RougeHandler

from .config import Config
from .data import PtrDataset, TextPair, RawDataProvider
from .model import PointerNetwork
from .utils import translate_logits, build_vocab

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, config: Config):

        self.config = config
        self.device = self.config.device
        
        (
            self.train_data_loader, 
            self.valid_data_loader, 
            self.test_data_loader,
            self.vocab,
        ) = self.prepare_data_and_vocab(self.config.raw_data_path)

        self.model = self.prepare_model(self.config, self.vocab, self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            amsgrad=True,
        )

        self.linear_scheduler = get_linear_schedule_with_warmup_ep(
            optimizer=self.optimizer,
            num_warmup_epochs=self.config.warmup_epochs,
            total_epochs=self.config.epoch,
            last_epoch=-1,
        )

        logger.info(
            "Trainer: Count parameters:{}".format(
                sum(p.nelement() for p in self.model.parameters()),
            ),
        )

        logger.info("Trainer: model and data initialized")

        self.global_train_step = 0
        self.global_valid_step = 0
        self.global_test_step = 0

        self.summary_writer = SummaryWriter(
            self.config.summary_writer_path,
        )

        self.loss_record_on_valid = []
        self.rouge_avg_record_on_valid = []
        self.train_record = []

    def prepare_model(self, config, vocab, device):
        model = PointerNetwork(config=config, vocab=vocab).to(device)
        model.apply(weight_init)
        
        return model
    
    def prepare_data_and_vocab(self):
        data_provider = RawDataProvider(self.config)
        train_set, valid_set, test_set, vocab = data_provider.load_raw_data(
            tokenizer=lambda x: list(x), create_vocab=True,
        )
        
        train_dataset = PtrDataset(self.config, vocab, train_set)
        valid_dataset = PtrDataset(self.config, vocab, valid_set)
        test_dataset = PtrDataset(self.config, vocab, test_set)
        
        train_data_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
        )
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=self.config.batch_size,
        )
        test_data_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
        )
        
        return train_data_loader, valid_data_loader, test_data_loader, vocab
    
    def forward_model(self, data, oov_count, phase):
        loss, logits, p_gens = self.model(
            src_token_ids=data["src_token_ids"],
            src_token_ids_with_oov=data["src_token_ids_with_oov"],
            tgt_token_ids=data["tgt_token_ids"],
            tgt_token_ids_with_oov=data["tgt_token_ids_with_oov"],
            src_valid_length=data["src_valid_len"],
            tgt_valid_length=data["tgt_valid_len"],
            oov_count=oov_count,
            phase=phase,
        )
        return loss, logits, p_gens

    def iteration(self, epoch, data_loader, phase: Phase):

        data_iter = tqdm(
            enumerate(data_loader),
            desc="EP:{}:{}".format(phase.name, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        total_loss = []

        curr_epoch_real_text_list = []
        curr_epoch_pred_text_list = []

        for idx, data in data_iter:
            oov_mapper = data_loader.dataset.oov_mapper
            oov_count_of_this_batch = len(oov_mapper)
            reverse_oov_mapper = {v: k for k, v in oov_mapper.items()}

            if phase == Phase.TRAIN:
                self.global_train_step += 1
            elif phase == Phase.VALID:
                self.global_valid_step += 1
            else:
                self.global_test_step += 1

            # data to device
            data = {key: value.to(self.device) for key, value in data.items()}
            
            # forward the model
            if phase == Phase.TRAIN:
                loss, logits, p_gens = self.forward_model(
                    data, oov_count=oov_count_of_this_batch, phase=phase,
                )

            else:
                # 此处主要是为了记录验证集和测试集的loss和p_gen
                with torch.no_grad():
                    loss, logits, p_gens = self.forward_model(
                        data, oov_count=oov_count_of_this_batch, phase=phase,
                    )

            # 对验证集计算rouge指标
            if phase is Phase.VALID:
                # global beam search
                beam_search_result = self.model.forward_beam_search(
                    src_token_ids=data["src_token_ids"],
                    src_token_ids_with_oov=data["src_token_ids_with_oov"],
                    src_valid_length=data["src_valid_len"],
                    oov_count=oov_count_of_this_batch,
                    SOS=self.vocab.sos_idx,
                    EOS=self.vocab.eos_idx,
                    max_decode_len=self.config.max_decode_len,
                    beam_size=self.config.beam_size,
                    top_k=self.config.decode_top_k,
                    sample_count=-1,
                )

                src_samples = translate_logits(
                    idx2str=self.vocab.idx2str,
                    oov_mapper=reverse_oov_mapper,
                    token_ids=data["src_token_ids_with_oov"],
                    num=-1,
                )

                tgt_samples = translate_logits(
                    idx2str=self.vocab.idx2str,
                    oov_mapper=reverse_oov_mapper,
                    token_ids=data["tgt_token_ids_with_oov"],
                    num=-1,
                )

                samples = pad_sequence(
                    [i[0]["token_ids"] for i in beam_search_result],
                ).transpose(0, 1)

                gen_samples = translate_logits(
                    idx2str=self.vocab.idx2str,
                    oov_mapper=reverse_oov_mapper,
                    token_ids=samples,
                    num=-1,
                )

                # 记录新beam search的结果
                curr_epoch_pred_text_list.extend(gen_samples)
                curr_epoch_real_text_list.extend(tgt_samples)

                print("\n======= Decoding samples =======\n")
                cnt = 1
                for s, t, g in zip(src_samples, tgt_samples, gen_samples):
                    print(f"-= {cnt} =-")
                    cnt += 1
                    print(
                        f"**SRC**: {' '.join(s)}\n\n"
                        f"**TGT**: {' '.join(t)}\n\n"
                        f"**GEN**: {' '.join(g)}\n\n",
                    )
                    if cnt == self.config.beam_search_sample_count:
                        break
                print("================================\n")

            # clear the oov mapper
            data_loader.dataset.fetch_and_clear_mapper()

            total_loss.append(loss.item())

            # do backward if on train
            if phase == Phase.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()

                if self.config.gradient_clip:
                    clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip,
                    )
                self.optimizer.step()

            log_info = {
                "phase": phase.name,
                "epoch": epoch,
                "iter": idx,
                "curr_loss": loss.item(),
                "avg_loss": sum(total_loss) / len(total_loss),
                "p_gen": sum(p_gens) / len(p_gens),
            }

            self.handle_summary(phase, log_info)

            if self.config.verbose and not idx % self.config.verbose:
                data_iter.write(str(log_info))

        if phase == Phase.TRAIN:
            self.linear_scheduler.step()  # step every train epoch
            self.summary_writer.add_scalar(
                "LR/learning_rate",
                self.linear_scheduler.get_last_lr()[0],
                epoch,
            )

        avg_loss = sum(total_loss) / len(total_loss)

        if phase is Phase.VALID:
            # calc rouge
            curr_epoch_real_text_list = list(
                map(
                    lambda x: " ".join(x), curr_epoch_real_text_list,
                ),
            )
            curr_epoch_pred_text_list = list(
                map(
                    lambda x: " ".join(x), curr_epoch_pred_text_list,
                ),
            )
            
            rouge_score = RougeHandler.batch_get_rouge_score(
                generated_list=curr_epoch_pred_text_list,
                ground_truth_list=curr_epoch_real_text_list,
            )
        else:
            rouge_score = None

        logger.info(
            "EP:{}_{}, avg_loss={}, rouge={}".format(
                epoch,
                phase.name,
                avg_loss,
                rouge_score,
            ),
        )

        # 记录训练信息
        record = {
            "epoch": epoch,
            "status": phase.name,
            "avg_loss": avg_loss,
            "rouge": rouge_score,
        }

        self.train_record.append(record)

        # handle rouge summary
        if phase is Phase.VALID:
            self.summary_writer.add_scalar(
                f"{phase.name}/rouge_avg/f",
                record["rouge"]["rouge-avg"]["f"],
                epoch,
            )

            self.summary_writer.add_scalar(
                f"{phase.name}/rouge_avg/p",
                record["rouge"]["rouge-avg"]["p"],
                epoch,
            )

            self.summary_writer.add_scalar(
                f"{phase.name}/rouge_avg/r",
                record["rouge"]["rouge-avg"]["r"],
                epoch,
            )

        # check should early stopping at valid
        if phase == Phase.VALID:
            self.loss_record_on_valid.append(avg_loss)
            self.rouge_avg_record_on_valid.append(
                rouge_score["rouge-avg"]["f"],
            )

            should_stop = check_should_do_early_stopping(
                self.rouge_avg_record_on_valid,
                self.config.not_early_stopping_at_first,
                self.config.es_with_no_improvement_after,
                acc_like=True,
                verbose=True,
            )

            if should_stop:
                best_epoch = should_stop
                logger.info("Now stop training..")
                return best_epoch
        return False

    def train(self, epoch):
        self.model.train()
        return self.iteration(epoch, self.train_data_loader, phase=Phase.TRAIN)

    def valid(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self.valid_data_loader, phase=Phase.VALID)

    def test(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self.test_data_loader, phase=Phase.TEST)

    def save_state_dict(self, epoch, save_to):
        output_path = save_to + ".ep{}".format(epoch)
        torch.save(self.model.state_dict(), output_path)

        logger.info("EP:{} model saved to {}".format(epoch, output_path))
        return output_path

    def load_state_dict(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(
                torch.load(
                    model_path, map_location=torch.device('cpu'),
                ),
            )
        logger.info("Model loaded from: {}".format(model_path))

    def start_train(self):
        # start training
        try:
            for epoch in range(self.config.epoch):
                self.train(epoch)
                self.save_state_dict(epoch, self.config.model_path)

                early_stop = self.valid(epoch)
                if early_stop:
                    break

                self.test(epoch)

            for record in self.train_record:
                logger.info(record)
                logger.info("\n")

            if early_stop:
                return early_stop

        except KeyboardInterrupt:
            logger.info(
                "Early stopping by KeyboardInterrupt, "
                "training record:",
            )

            for record in self.train_record:
                logger.info(record)
                logger.info("\n")

    def get_metric_score(self, output, label, threshold=0.5):
        assert output.shape == label.shape
        output = (output > threshold).to(torch.long)

        correct = (output == label).sum().item()
        total = label.numel()

        accuracy = correct / total

        return correct, total, accuracy

    def handle_summary(self, phase: Phase, log_info: dict):
        if phase is Phase.TRAIN:
            step_count = self.global_train_step
        elif phase is Phase.VALID:
            step_count = self.global_valid_step
        else:
            step_count = self.global_test_step

        status = phase.name

        self.summary_writer.add_scalar(
            f"{status}/curr/loss", log_info["curr_loss"], step_count,
        )

        self.summary_writer.add_scalar(
            f"{status}/avg/loss", log_info["avg_loss"], step_count,
        )

        self.summary_writer.add_scalar(
            f"P_gen/{status}", log_info["p_gen"], step_count,
        )


class Tester(object):

    def __init__(self, config: Config):

        self.config = config
        self.device = self.config.device

        self.vocab = Vocab(config)
        self.model = PointerNetwork(
            config=self.config, vocab=self.vocab,
        ).to(self.device)

        if not os.path.exists(self.config.inference_model_path):
            logger.error(
                f"Can't find ptr model at {self.config.inference_model_path},"
                "now exiting..",
            )
            exit(-1)

        self.model.load_state_dict(
            torch.load(
                self.config.inference_model_path,
                map_location=torch.device('cpu'),
            ),
        )
        self.model.eval()

        logger.info("Model and weight loaded")

    def do_beam_search(
        self, beam_size, top_k, data, oov_count_of_this_batch,
        reverse_oov_mapper,
    ):
        beam_search_result = self.model.forward_beam_search(
            src_token_ids=data["src_token_ids"],
            src_token_ids_with_oov=data["src_token_ids_with_oov"],
            src_valid_length=data["src_valid_len"],
            oov_count=oov_count_of_this_batch,
            SOS=self.vocab.sos_idx,
            EOS=self.vocab.eos_idx,
            max_decode_len=self.config.max_decode_len,
            beam_size=beam_size,
            top_k=top_k,
            sample_count=-1,
        )
        src_samples = translate_logits(
            idx2str=self.vocab.idx2str,
            oov_mapper=reverse_oov_mapper,
            token_ids=data["src_token_ids_with_oov"],
            num=-1,
        )
        tgt_samples = translate_logits(
            idx2str=self.vocab.idx2str,
            oov_mapper=reverse_oov_mapper,
            token_ids=data["tgt_token_ids_with_oov"],
            num=-1,
        )
        samples = pad_sequence(
            [i[0]["token_ids"] for i in beam_search_result],
        ).transpose(0, 1)

        gen_samples = translate_logits(
            idx2str=self.vocab.idx2str,
            oov_mapper=reverse_oov_mapper,
            token_ids=samples,
            num=-1,
        )
        return src_samples, tgt_samples, gen_samples

    def predict_dataloader(
        self, data_loader, calc_rouge=False, beam_size=None, top_k=None,
    ):

        src_text_list = []
        real_text_list = []
        pred_text_list = []

        for data in tqdm(data_loader):
            oov_mapper = data_loader.dataset.oov_mapper
            oov_count_of_this_batch = len(oov_mapper)
            reverse_oov_mapper = {v: k for k, v in oov_mapper.items()}

            data = {key: value.to(self.device) for key, value in data.items()}

            if beam_size is None:
                beam_size = self.config.beam_size

            if top_k is None:
                top_k = self.config.decode_top_k

            with torch.no_grad():
                src_samples, tgt_samples, gen_samples = (
                    self.do_beam_search(
                        beam_size, top_k, data, oov_count_of_this_batch,
                        reverse_oov_mapper,
                    )
                )

            src_text_list.extend(src_samples)
            real_text_list.extend(tgt_samples)
            pred_text_list.extend(gen_samples)

            data_loader.dataset.fetch_and_clear_mapper()

        src_text_list = list(map(lambda x: " ".join(x), src_text_list))
        real_text_list = list(map(lambda x: " ".join(x), real_text_list))
        pred_text_list = list(map(lambda x: " ".join(x), pred_text_list))

        rouge_scores = RougeHandler.batch_get_rouge_score(
            generated_list=pred_text_list,
            ground_truth_list=real_text_list,
        )

        src_text_list = batch_beatify_output(src_text_list)
        real_text_list = batch_beatify_output(real_text_list)
        pred_text_list = batch_beatify_output(pred_text_list)

        result = {
            "src_text_list": src_text_list,
            "real_text_list": real_text_list,
            "gen_text_list": pred_text_list,
            "rouge_scores": rouge_scores if calc_rouge else None,
        }

        return result

    def predict_text_list(
        self, text_list, real_text_list, beam_size=None, top_k=None,
        calc_rouge=False,
    ):
        if not real_text_list:
            real_text_list = [
                "1" * self.config.max_tgt_len for i in range(len(text_list))
            ]
            if calc_rouge:
                logger.warning(
                    "calc_rouge requires `real_text_list`!",
                )
                calc_rouge = False
        else:
            assert len(real_text_list) == len(text_list), (
                "Make sure that `text_list` and `real_text_list` has the "
                "same length."
            )

        text_pairs = [
            TextPair(source=src, simplified=tgt)
            for src, tgt in zip(text_list, real_text_list)
        ]

        dataset = PtrDataset(
            config=self.config,
            vocab=self.vocab,
            text_pairs=text_pairs,
        )

        data_loader = DataLoader(dataset, batch_size=self.config.batch_size)

        return self.predict_dataloader(
            data_loader=data_loader,
            calc_rouge=calc_rouge,
            beam_size=beam_size,
            top_k=top_k,
        )

    def predict_to_file(
        self, text_list, real_text_list, export_to, beam_size=None, top_k=None,
        calc_rouge=False,
    ):
        predict = self.predict_text_list(
            text_list, real_text_list, beam_size=beam_size, top_k=top_k,
            calc_rouge=calc_rouge,
        )

        with open(export_to, "w") as fwt:
            for k, v in predict["rouge_scores"].items():
                fwt.write(f"{k}:{v}\n")
            fwt.write("\n")

            cnt = 0

            for src, real, gen in zip(
                predict["src_text_list"],
                predict["real_text_list"],
                predict["gen_text_list"],
            ):
                fwt.write(
                    f"-= {cnt} =-\nsrc:{src}\nreal:{real}\ngen:{gen}\n\n",
                )
                cnt += 1
