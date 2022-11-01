import math

import torch
import numpy as np
import random
from tqdm import tqdm

# 0 - None
# 1 - Aspect-Begin
# 2 - Aspect-In
# 3 - Opinion-Begin
# 4 - Opinion-In
ASPECT_BEGIN=1
ASPECT_IN=2
OPINION_BEGIN=3
OPINION_IN=4
PAIR=5
sentiment2id = {'观点-负面': 1, '观点-中性':2, '观点-正面': 3}
from transformers import AutoTokenizer

class Instance(object):
    def __init__(self, tokenizer, sentence_pack, args):
        self.sentence = sentence_pack['text'].strip()
        if args.do_lower_case:
            self.sentence = self.sentence.lower()
        # add a special token acount for opinion with no aspect
        self.sentence = '## ' + self.sentence
        self.tokens = self.sentence.split()
        self.sen_length = len(self.tokens)
        self.token_range = []
        if 'roberta' in args.bert_tokenizer_path:
            self.bert_tokens = tokenizer.encode(self.tokens,is_split_into_words=True)
        else:
            self.bert_tokens = tokenizer.encode(self.sentence)
        self.length = len(self.bert_tokens)
        if self.length > args.max_sequence_len:
            self.valid = False
            return 
        self.valid = True
        self.bert_tokens_padding = torch.zeros(args.max_sequence_len).long()
        self.ner_tags = -1 * torch.ones(args.max_sequence_len).long()
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.mask = torch.zeros(args.max_sequence_len)

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        self.mask[:self.length] = 1

        token_start = 1
        for i, w, in enumerate(self.tokens):
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        assert self.length == self.token_range[-1][-1]+2

        # self.length = min(self.length,args.max_sequence_len)

        self.tags[:, :] = -1
        for i in range(1, self.length-1):
            self.ner_tags[i] = 0
            for j in range(i, self.length-1):
                self.tags[i][j] = 0

        def update_span(s):
            s = [x+1 for x in s]
            s[-1] -= 1
            return s

        for triple in sentence_pack['tags']:
            # 过滤只包含中性情感的opinion
            if 'aspect' not in triple and triple['sentiment'] == '观点-中性':
                continue
            if 'aspect' not in triple:
                aspect_span = [[0,0]]
            else:
                aspect = triple['aspect']
                if self.tokens[aspect[-1]] in '.!?;,':
                    aspect = (aspect[0],aspect[1]-1)
                aspect_span = [update_span(aspect)]
            if 'opinion' not in triple:
                continue
            # 去除后面的标点符号
            opinion = triple['opinion']
            if self.tokens[opinion[-1]] in '.!?;,':
                opinion = (opinion[0],opinion[1]-1)
            opinion_span = [update_span(triple['opinion'])]

            '''set tag for aspect'''
            for l, r in aspect_span:
                start = self.token_range[l][0]
                # end = self.token_range[r][1]
                # for i in range(start, end+1):
                #     for j in range(i, end+1):
                #         self.tags[i][j] = ASPECT_IN
                # self.tags[start][start] = ASPECT_BEGIN
                for i in range(l, r+1):
                    al, ar = self.token_range[i]
                    self.ner_tags[al] = ASPECT_IN
                    self.ner_tags[al+1:ar+1] = -1
                    # '''mask positions of sub words'''
                    # self.tags[al+1:ar+1, :] = -1
                    # self.tags[:, al+1:ar+1] = -1
                self.ner_tags[start] = ASPECT_BEGIN

            '''set tag for opinion'''
            for l, r in opinion_span:
                start = self.token_range[l][0]
                # end = self.token_range[r][1]
                # for i in range(start, end+1):
                #     for j in range(i, end+1):
                #         self.tags[i][j] = OPINION_IN
                # self.tags[start][start] = OPINION_BEGIN
                for i in range(l, r+1):
                    pl, pr = self.token_range[i]
                    self.ner_tags[pl] = OPINION_IN
                    self.ner_tags[pl+1:pr+1] = -1
                    # self.tags[pl+1:pr+1, :] = -1
                    # self.tags[:, pl+1:pr+1] = -1
                self.ner_tags[start] = OPINION_BEGIN

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal:sar+1, spl:spr+1] = -1
                            if args.task == 'pair':
                                if i > j:
                                    self.tags[spl][sal] = PAIR
                                else:
                                    self.tags[sal][spl] = PAIR
                            elif args.task == 'triplet':
                                if i > j:
                                    self.tags[spl][sal] = sentiment2id[triple['sentiment']]
                                else:
                                    self.tags[sal][spl] = sentiment2id[triple['sentiment']]


def load_data_instances(sentence_packs, args):
    instances = list()
    if 'roberta' in args.bert_tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_path,add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_path)
    print('Raw data size',len(sentence_packs))
    texts = set()
    dup = 0
    for sentence_pack in tqdm(sentence_packs):
        text = sentence_pack['text']
        if text in texts:
            dup += 1
            continue
        texts.add(text)
        instance = Instance(tokenizer, sentence_pack, args)
        if instance.valid:
            instances.append(instance)
    # print(tokenizer.convert_ids_to_tokens(instances[3].bert_tokens))
    print('dup',dup)
    print('Processed data size',len(instances))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def shuffle(self):
        random.shuffle(self.instances)

    def get_batch(self, index):
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        tags = []
        ner_tags = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            tags.append(self.instances[i].tags)
            ner_tags.append(self.instances[i].ner_tags)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)
        ner_tags = torch.stack(ner_tags).to(self.args.device)
        return bert_tokens, lengths, masks, sens_lens, token_ranges, tags, ner_tags
