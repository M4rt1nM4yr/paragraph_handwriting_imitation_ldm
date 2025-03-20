import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.utils.constants import *
from src.data.utils.subsequent_mask import subsequent_mask


def custom_collate(batch):
    imgs_in = []
    texts_in = []
    names_in = []
    text_logits_ctc_in = []
    text_logits_s2s_in = []
    texts_len = []
    writers = []
    key_padding = []
    style_samples = []
    style_padding = []
  #  bin = Binarize()


    for item in batch:
      #  imgs_in.append(bin(item[IMAGE]))
        imgs_in.append(item[IMAGE])
        texts_in.append(item[TEXT])
        names_in.append(item["name"])
        if TEXT_LOGITS_S2S in item.keys():
            text_logits_s2s_in.append(item[TEXT_LOGITS_S2S])
        text_logits_ctc_in.append(item[TEXT_LOGITS_CTC])
        texts_len.append(len(item[TEXT]))
        writers.append(item[WRITER])

        if item[SRC_KEY_PADDING] is not None:
            key_padding.append(item[SRC_KEY_PADDING])

        if item[STYLE_SAMPLE] is not None:
            style_samples.append(item[STYLE_SAMPLE])

        if item[STYLE_PADDING] is not None:
            style_padding.append(item[STYLE_PADDING])



    imgs = torch.stack(imgs_in)
    key_padding_masks = None
    if len(key_padding) > 0:
        key_padding_masks = torch.stack(key_padding)

    style_samples_torch = None
    if len(style_samples) > 0:
        style_samples_torch = torch.stack(style_samples)

    style_padding_torch = None
    if len(style_padding) > 0:
        style_padding_torch = torch.stack(style_padding)

    text_logits_ctc_out = pad_sequence(text_logits_ctc_in, padding_value=1, batch_first=True)

    out_dict = {
            IMAGE: imgs,
            TEXT: texts_in,
            TEXT_LOGITS_CTC: text_logits_ctc_out,
            UNPADDED_TEXT_LEN: torch.LongTensor(texts_len),
            WRITER: torch.LongTensor(writers),
            SRC_KEY_PADDING: key_padding_masks,
            STYLE_SAMPLE: style_samples_torch,
            STYLE_PADDING: style_padding_torch,
            "name": names_in,

    }


    if len(text_logits_s2s_in)>0:
        text_logits_s2s_out = pad_sequence(text_logits_s2s_in, padding_value=1, batch_first=True)
        out_dict[TEXT_LOGITS_S2S] = text_logits_s2s_out
        out_dict[TGT_KEY_PADDING_MASK] = torch.eq(text_logits_s2s_out,
                              torch.ones(text_logits_s2s_out.shape, dtype=torch.long) * torch.LongTensor([1]))
        out_dict[TGT_MASK] = subsequent_mask(text_logits_s2s_out.shape[-1] - 1)

    return out_dict