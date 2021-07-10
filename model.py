import logging
import random
from queue import Queue

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import Config, Phase
from .utils import Vocab

logger = logging.getLogger(__name__)


class Encoder(nn.Module):

    def __init__(self, config: Config, vocab: Vocab):

        super(Encoder, self).__init__()
        self.config = config
        self.vocab = vocab

        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.fc_output = nn.Linear(
            in_features=self.config.hidden * 2,
            out_features=self.config.hidden * 2,
            bias=False,
        )

        self.fc_output_feature = nn.Linear(
            in_features=self.config.hidden * 2,
            out_features=self.config.hidden * 2,
        )

        self.fc_hn = nn.Linear(
            self.config.hidden * 2,
            self.config.hidden,
        )

        self.fc_cn = nn.Linear(
            self.config.hidden * 2,
            self.config.hidden,
        )

        self.relu = nn.ReLU()

    def forward(self, source_token_ids, valid_length):
        """Forward the model.

        Args:
            source_token_ids (Tensor): token_ids of source text,
                shape: [bs, max_src_len]
            valid_length (Tensor): valid length of src text,
                shape: [bs]
        """
        bs = source_token_ids.shape[0]

        source_repr = self.embedding(source_token_ids)  # bs, max_src_len, E

        packed_repr = pack_padded_sequence(
            source_repr, valid_length.cpu().to(torch.int64), batch_first=True,
            enforce_sorted=False,
        )

        # output: [bs, max_src_len, hidden * 2]
        # hn, cn: [2, bs, hidden]
        output, (hn, cn) = self.lstm(packed_repr)

        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=self.config.max_src_len,
        )

        # bs, max_src_len, hidden * 2
        output = self.fc_output(output)

        output_feature = self.fc_output_feature(output)

        hn = hn.transpose(0, 1).contiguous().view(bs, -1)  # 2, bs, hidden
        cn = cn.transpose(0, 1).contiguous().view(bs, -1)  # 2, bs, hidden

        hn = self.relu(self.fc_hn(hn)).unsqueeze(0)  # 1, bs, hidden
        cn = self.relu(self.fc_cn(cn)).unsqueeze(0)  # 1, bs, hidden

        return output, output_feature, hn, cn, source_repr


class Attention(nn.Module):

    def __init__(self, config: Config):
        super(Attention, self).__init__()
        self.config = config

        self.w_c = nn.Linear(1, self.config.hidden * 2, bias=False)
        self.w_h = nn.Linear(
            self.config.hidden * 2,
            self.config.hidden * 2,
            bias=False,
        )  # for encoder hidden state

        self.w_s = nn.Linear(
            self.config.hidden * 2,
            self.config.hidden * 2,
            bias=True,
        )  # for decoder hidden state

        self.v = nn.Linear(self.config.hidden * 2, 1, bias=False)

    def forward(self, encoder_output, decoder_hidden, mask, coverage):
        encoder_feature = self.w_h(encoder_output)  # bs, max_len, hidden
        decoder_feature = self.w_s(decoder_hidden)  # bs, 1, hidden

        # bs, max_src_len, hidden
        attention_features = encoder_feature + decoder_feature

        if self.config.use_coverage:
            # bs, max_src_len, 2 * hidden
            coverage_feature = self.w_c(coverage.unsqueeze(-1))

            attention_features = attention_features + coverage_feature

        # bs, max_src_len, 2 * hidden
        attention_features = torch.tanh(attention_features)

        # bs, max_src_len
        scores = self.v(attention_features).squeeze(-1)

        # normalize score
        scores = torch.softmax(scores, dim=-1)

        # bs, max_src_len
        attention_distribution = scores * mask

        # bs, 1
        normalize_factor = attention_distribution.sum(dim=-1, keepdim=True)

        # bs, max_src_len
        attention_distribution = attention_distribution / normalize_factor

        # calc attention context: [bs, hidden]
        context = torch.bmm(
            attention_distribution.unsqueeze(1),
            encoder_output,
        ).squeeze(1)

        if self.config.use_coverage:
            # update coverage: bs, src_max_len
            coverage = coverage + attention_distribution.squeeze(1)

        return context, attention_distribution, coverage


class Decoder(nn.Module):

    def __init__(self, config: Config, encoder: Encoder):
        super(Decoder, self).__init__()

        self.config = config
        self.encoder = encoder

        self.attention = Attention(config)

        self.lstm = nn.LSTM(
            self.config.embedding_dim, self.config.hidden,
            bidirectional=False, batch_first=True,
        )

        self.fc_merge_input_cnt_context = nn.Linear(
            self.config.decoder_input_dim + self.config.hidden * 2,
            self.config.decoder_input_dim,
            bias=True,
        )

        # for p_gen:
        self.fc_p_gen = nn.Linear(self.config.hidden * 4, 1, bias=True)

        self.fc1 = nn.Linear(self.config.hidden * 3, self.config.hidden)
        self.fc2 = nn.Linear(self.config.hidden, self.config.vocab_size)
        self.relu = nn.ReLU()

    def create_mask(self, input_, valid_length):
        bs, max_len, embedding_dim = input_.shape
        mask = torch.tensor(
            range(max_len),
            device=self.config.device,
            dtype=torch.long,
        ).view(1, max_len).repeat(bs, 1)
        valid_length = valid_length.view(bs, -1).repeat(1, max_len)
        mask = mask.lt(valid_length)

        return mask

    def get_initial_input(self, batch_size):
        """Repeat dummy input of <SOS> along the batch size dimension.

        Args:
            batch_size (int): batch size

        Returns:
            Tensor: initial input of decoder: [bs, 1, input_hidden]
        """
        sos_idx = torch.tensor(
            self.encoder.vocab.sos_idx, device=self.config.device,
        )

        sos_embedding = self.encoder.embedding(sos_idx).view(1, 1, -1)
        sos_embedding = sos_embedding.repeat(batch_size, 1, 1)
        return sos_embedding

    def decode_one_step(
        self, token_ids_with_unk, decoder_input, decoder_hidden,
        encoder_out_feature, padding_mask, context, coverage, extend_zeros,
    ):
        # hidden = decoder_hidden.view(1, bs, -1)
        hidden = decoder_hidden

        # concat input and context, then pass into a linear layer
        input_ = self.fc_merge_input_cnt_context(
            torch.cat([decoder_input, context.unsqueeze(1)], dim=-1),
        )  # bs, 1, decoder_input_dim

        # bs, 1, hidden
        output_, hidden = self.lstm(input_, hidden)

        # h, c: [1, bs, hidden]
        h, c = hidden

        # [1, bs, 2 * hidden]
        merged_hidden = torch.cat([h, c], dim=-1)

        # [bs, 1, 2 * hidden]
        merged_hidden = merged_hidden.transpose(0, 1)

        # context: bs, hidden
        # scores: bs, 1, src_max_len
        context, scores, coverage = self.attention(
            encoder_output=encoder_out_feature,
            decoder_hidden=merged_hidden,
            mask=padding_mask,
            coverage=coverage,
        )

        # bs, hidden * 2
        concat_output_context = torch.cat(
            [output_.squeeze(1), context], dim=-1,
        )

        p_gen = torch.sigmoid(
            self.fc_p_gen(
                torch.cat(
                    [input_.squeeze(1), context, output_.squeeze(1)], dim=-1,
                ),
            ),
        )

        # two linear layer...
        output = self.fc2(self.fc1(concat_output_context))

        vocab_distribution = torch.softmax(output, dim=-1)

        if self.config.use_ptr:
            # bs * vocab_size
            vocab_distribution = vocab_distribution * p_gen

            # bs * max_src_len
            attention_distribution = scores.squeeze(1) * (1 - p_gen)

            # extended vocab: bs, vocab_size + oov_count
            final_distribution = torch.cat(
                [vocab_distribution, extend_zeros], dim=-1,
            )
            final_distribution.scatter_add_(
                dim=1, index=token_ids_with_unk, src=attention_distribution,
            )

        else:
            final_distribution = vocab_distribution

        # get max_indices for the next decoder input: [bs,]
        max_indices = final_distribution.argmax(-1)

        return (
            final_distribution, hidden, context, attention_distribution,
            max_indices, p_gen, coverage,
        )

    def beam_search(
        self, src_token_ids_with_oov, encoder_output, decoder_hidden,
        encoder_mask, oov_count, SOS, EOS, max_decode_len, beam_size, top_k,
        sample_count,
    ):
        """Perform beam search on multi-batch logits.

        Args:
            logits (Tensor): logits of decoder: bs, max_decode_len, vocab_size
            SOS (int): token id of start-of-sentence token
            EOS (int): token id of end-of-sentence token
            max_decode_len (int): max decode length
            beam_size (int, optional): size of beam search. Defaults to 1.
            top_k (int, optional): how many candidates to reserve at the end
                of the decoding process. Defaults to 1.

        Returns:
            List[List[Dict]]: decoding result of every batchs, each contains
            top_k candidates,
                candidate: {"token_ids": torch.Tensor, "prob": torch.Tensor}
        """
        max_depth = max_decode_len  # max_decode_length

        bs, *_ = encoder_output.shape

        # will be added to original vocab distribution
        extend_zeros = torch.zeros(
            bs, oov_count, device=encoder_output.device, dtype=torch.float,
        )

        all_batch_results = []

        if sample_count == -1:
            sample_count = bs

        # run beam search over batch i
        for i in range(sample_count):

            terminus_nodes = []
            node_queue = Queue()

            hn, cn = decoder_hidden
            curr_hn = hn[:, i, :].unsqueeze(1)
            curr_cn = cn[:, i, :].unsqueeze(1)

            # dummy node
            root = BeamSearchNode(
                previous_node=None,
                token_id=torch.tensor(SOS),
                log_prob=0,
                depth=0,
                hidden=(curr_hn, curr_cn),
                context=torch.zeros(
                    1, self.config.hidden * 2, device=self.config.device,
                ),
                coverage=torch.zeros(
                    1, self.config.max_src_len, device=self.config.device,
                ),
            )

            node_queue.put(root)

            while not node_queue.empty():

                candidates = []

                for _ in range(node_queue.qsize()):
                    node = node_queue.get()

                    if node.token_id == EOS or node.depth >= max_depth:
                        terminus_nodes.append(
                            (node, node.get_sequence_prob()),
                        )

                        # ? do we need this ?
                        # if len(terminus_nodes) == beam_size:
                        #     node_queue.queue.clear()
                        #     break
                        continue

                    node_idx = node.token_id.to(self.config.device)
                    node_idx = node_idx.view(1, 1)

                    curr_hn = node.hidden[0].to(self.config.device)
                    curr_cn = node.hidden[1].to(self.config.device)
                    context = node.context.to(self.config.device)
                    coverage = node.coverage.to(self.config.device)

                    decoder_input = self.encoder.embedding(node_idx)

                    curr_src_token_ids_with_oov = (
                        src_token_ids_with_oov[i].unsqueeze(0)
                    )
                    curr_encoder_output = encoder_output[i].unsqueeze(0)
                    curr_encoder_mask = encoder_mask[i].unsqueeze(0)
                    curr_extend_zeros = extend_zeros[i].unsqueeze(0)

                    final_distribution, curr_hidden, context, *_, coverage = (
                        self.decode_one_step(
                            token_ids_with_unk=curr_src_token_ids_with_oov,
                            decoder_input=decoder_input,
                            decoder_hidden=(curr_hn, curr_cn),
                            encoder_out_feature=curr_encoder_output,
                            padding_mask=curr_encoder_mask,
                            context=context,
                            coverage=coverage,
                            extend_zeros=curr_extend_zeros,
                        )
                    )

                    # [extend_vocab_len,]
                    curr_logits = final_distribution.squeeze(0)

                    curr_logits = torch.log(curr_logits)

                    log_prob, indices = curr_logits.topk(
                        beam_size + 1, dim=-1,
                    )

                    cnt = 0
                    for j in range(beam_size + 1):
                        prob_ = log_prob[j]
                        index = indices[j]

                        # skip UNK when decoding
                        if index.cpu().item() == self.encoder.vocab.unk_idx:
                            continue

                        # restrain the token index in the vocab(not extended)
                        # otherwise we can't lookup the embedding.
                        raw_index = index.clone()
                        index = index.masked_fill(
                            index >= self.encoder.vocab.size,
                            self.encoder.vocab.unk_idx,
                        )

                        child = BeamSearchNode(
                            previous_node=node,
                            token_id=index,
                            log_prob=prob_ + node.log_prob,
                            depth=node.depth + 1,
                            hidden=curr_hidden,
                            context=context,
                            coverage=coverage,
                            raw_token_id=raw_index,
                        )

                        score = child.get_sequence_prob()
                        candidates.append((child, score))

                        cnt += 1
                        if cnt == beam_size:
                            # no UNK in topk
                            break

                candidates.sort(key=lambda x: x[1], reverse=True)

                for k in range(min(beam_size, len(candidates))):
                    node_queue.put(candidates[k][0])

            # now we have all top-k candidate paths, do backtrack
            terminus_nodes = sorted(
                terminus_nodes, key=lambda x: x[1], reverse=True,
            )

            top_k_results = []

            for node, _ in terminus_nodes[:top_k]:
                curr_top_k = {
                    "token_ids": [],
                    "prob": node.get_sequence_prob(),
                }

                token_ids = []

                while node:
                    if node.raw_token_id:
                        token_ids.append(node.raw_token_id)
                    else:
                        token_ids.append(node.token_id)

                    node = node.previous_node

                token_ids = [t.cpu() for t in token_ids]

                curr_top_k["token_ids"] = torch.stack(token_ids[::-1], dim=0)
                top_k_results.append(curr_top_k)

            all_batch_results.append(top_k_results)

        return all_batch_results

    def forward(
        self, token_ids_with_oov, encoder_output, decoder_hidden,
        tgt_token_ids, tgt_token_ids_with_oov, encoder_mask, decoder_mask,
        decoder_valid_len, oov_count, phase,
    ):

        bs, *_ = encoder_output.shape

        batch_loss = []
        all_distributions = []

        # will be add to original vocab distribution
        extend_zeros = torch.zeros(
            bs, oov_count, device=encoder_output.device, dtype=torch.float,
        )

        max_tgt_len = torch.max(decoder_valid_len)

        p_gens = []

        for idx in range(min(max_tgt_len, self.config.max_decode_len)):
            if idx == 0:
                # create initial input token <S>
                decoder_input = self.get_initial_input(bs)

                # do not train this initial attn context vector
                context = torch.zeros(
                    bs, self.config.hidden * 2, device=self.config.device,
                )

                # create coverage
                coverage = torch.zeros(
                    bs, self.config.max_src_len, device=self.config.device,
                )

            final_distribution, hidden, context, attention_distribution, \
                max_indices, p_gen, coverage = self.decode_one_step(
                    token_ids_with_unk=token_ids_with_oov,
                    decoder_input=decoder_input,
                    decoder_hidden=decoder_hidden,
                    encoder_out_feature=encoder_output,
                    padding_mask=encoder_mask,
                    context=context,
                    coverage=coverage,
                    extend_zeros=extend_zeros,
                )

            all_distributions.append(final_distribution)

            p_gens.extend(p_gen.squeeze(1).cpu().tolist())

            curr_target = tgt_token_ids_with_oov[:, idx]

            # probs of all samples of this batch at current time step
            probs = torch.gather(
                final_distribution, 1, curr_target.unsqueeze(1),
            )

            curr_loss = - torch.log(probs + 1e-12)

            if self.config.use_coverage:
                # add coverage loss
                coverage_loss = torch.sum(
                    torch.min(attention_distribution, coverage),
                )
                curr_loss += coverage_loss

            curr_loss = curr_loss * decoder_mask[:, idx].view(bs, -1)

            batch_loss.append(curr_loss)

            # update decoder input and hidden for next time step
            decoder_hidden = hidden

            if phase == Phase.TRAIN and self.config.teacher_forcing and (
                random.random() < self.config.teacher_forcing
            ):
                # 使用真实标签
                curr_tgt_token_ids = tgt_token_ids[:, idx]

                # bs, 1, hidden
                decoder_input = self.encoder.embedding(
                    curr_tgt_token_ids,
                ).unsqueeze(1)

            else:
                # bs, 1
                max_indices = max_indices.masked_fill(
                    max_indices >= self.encoder.vocab.size,
                    self.encoder.vocab.unk_idx,
                ).unsqueeze(-1)

                # bs, 1, hidden
                decoder_input = self.encoder.embedding(max_indices)

        loss_sum = torch.sum(torch.stack(batch_loss, dim=1), dim=1)

        loss_sum = loss_sum / decoder_valid_len.unsqueeze(-1)
        loss = torch.mean(loss_sum)

        # bs, max_decode_length, (vocab_size+oov_count)
        logits = torch.stack(all_distributions, dim=1)

        return loss, logits, p_gens


class BeamSearchNode(object):

    def __init__(
        self, previous_node, token_id, log_prob, depth, hidden=None,
        context=None, coverage=None, raw_token_id=None,
    ):
        self.previous_node = previous_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.depth = depth
        self.hidden = hidden
        self.context = context
        self.coverage = coverage
        self.raw_token_id = raw_token_id
        self.eps = 1e-9

    def get_sequence_prob(self, reward=1.0):
        return self.log_prob / ((self.depth + self.eps) * reward)


class PointerNetwork(nn.Module):

    def __init__(self, config: Config, vocab: Vocab):
        super(PointerNetwork, self).__init__()
        self.config = config
        self.vocab = vocab

        self.encoder = Encoder(config, vocab)
        self.decoder = Decoder(config, self.encoder)

    def create_mask(self, input_, valid_length):
        bs, max_len = input_.shape

        mask = torch.tensor(
            range(max_len),
            device=valid_length.device,
            dtype=valid_length.dtype,
        ).view(1, max_len).repeat(bs, 1)

        valid_length = valid_length.view(bs, -1).repeat(1, max_len)
        mask = mask.lt(valid_length + 1)

        return mask

    def forward(
        self, src_token_ids, src_token_ids_with_oov, tgt_token_ids,
        tgt_token_ids_with_oov, src_valid_length, tgt_valid_length,
        oov_count, phase,
    ):

        output, out_feature, hn, cn, source_repr = self.encoder(
            src_token_ids, src_valid_length,
        )

        src_padding_mask = self.create_mask(src_token_ids, src_valid_length)
        tgt_padding_mask = self.create_mask(tgt_token_ids, tgt_valid_length)

        loss, logits, p_gens = self.decoder(
            token_ids_with_oov=src_token_ids_with_oov,
            encoder_output=out_feature,
            decoder_hidden=(hn, cn),
            tgt_token_ids=tgt_token_ids,
            tgt_token_ids_with_oov=tgt_token_ids_with_oov,
            encoder_mask=src_padding_mask,
            decoder_mask=tgt_padding_mask,
            decoder_valid_len=tgt_valid_length,
            oov_count=oov_count,
            phase=phase,
        )

        return loss, logits, p_gens

    def forward_beam_search(
        self, src_token_ids, src_token_ids_with_oov, src_valid_length,
        oov_count, SOS, EOS, max_decode_len, beam_size, top_k,
        sample_count,
    ):

        output, out_feature, hn, cn, source_repr = self.encoder(
            src_token_ids, src_valid_length,
        )

        src_padding_mask = self.create_mask(src_token_ids, src_valid_length)

        beam_search_result = self.decoder.beam_search(
            src_token_ids_with_oov=src_token_ids_with_oov,
            encoder_output=out_feature,
            decoder_hidden=(hn, cn),
            encoder_mask=src_padding_mask,
            oov_count=oov_count,
            SOS=SOS,
            EOS=EOS,
            max_decode_len=max_decode_len,
            beam_size=beam_size,
            top_k=top_k,
            sample_count=sample_count,
        )

        return beam_search_result
