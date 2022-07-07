# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Text iterator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import random
import math
import jieba


class TextIterator:
    """Create text iterator for sequence pair classification problem. 
    Data file is assumed to contain one sample per line. The format is 
    label\tsequence1\tsequence2. 
    Args:
        input_file: path of the input text file.
        token_to_idx: a dictionary, which convert token to index
        batch_size: mini-batch size 
        vocab_size: limit on the size of the vocabulary, if token index is 
            larger than vocab_size, return UNK (index 1)
        shuffle: Boolean; if true, we will first sort a buffer of samples by 
            sequence length, and then shuffle it by batch-level.
        factor: buffer size is factor * batch-size 

    """

    def __init__(self, input_file, tokenizer,
                 batch_size=128, vocab_size=-1, shuffle=True, factor=20,
                 bos_token="[CLS]", eos_token="[SEP]"):
        self.input_file = open(input_file, "r", encoding="utf-8")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.end_of_data = False
        self.instance_buffer = []
        self.bos_token = bos_token
        self.eos_token = eos_token
        # buffer for shuffle
        self.max_buffer_size = batch_size * factor

    def __iter__(self):
        return self

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        instance = []

        if len(self.instance_buffer) == 0:
            for _ in range(self.max_buffer_size):
                line = self.input_file.readline()
                if line == "":
                    break
                # arr = line.strip("\n").split('\t')
                # assert len(arr) == 3
                # self.instance_buffer.append(arr)
                label, src_id, src_title, src_pvs, tgt_id, tgt_title, tgt_pvs = line.strip("\n").split('\t')
                src_text = src_title + " [SEP] " + " ".join(jieba.cut(src_pvs))
                tgt_text = tgt_title + " [SEP] " + " ".join(jieba.cut(tgt_pvs))
                self.instance_buffer.append((src_text, tgt_text, label))

            if self.shuffle:
                # sort by length of sum of target buffer and target_buffer
                length_list = []
                for ins in self.instance_buffer:
                    current_length = len(ins[1]) + len(ins[2])
                    length_list.append(current_length)

                length_array = numpy.array(length_list)
                length_idx = length_array.argsort()
                # shuffle mini-batch
                tindex = []
                small_index = list(range(
                    int(math.ceil(len(length_idx) * 1. / self.batch_size))))
                random.shuffle(small_index)
                for i in small_index:
                    if (i + 1) * self.batch_size > len(length_idx):
                        tindex.extend(length_idx[i * self.batch_size:])
                    else:
                        tindex.extend(
                            length_idx[i * self.batch_size:(i + 1) * self.batch_size])

                _buf = [self.instance_buffer[i] for i in tindex]
                self.instance_buffer = _buf

        if len(self.instance_buffer) == 0:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    current_instance = self.instance_buffer.pop(0)
                except IndexError:
                    break

                # sent1 = " ".join(jieba.cut(current_instance[0]))
                # sent2 = " ".join(jieba.cut(current_instance[1]))
                sent1 = current_instance[0]
                sent2 = current_instance[1]
                label = int(current_instance[2])

                sent1_tokens = [self.bos_token] + self.tokenizer.tokenize(sent1) + [self.eos_token]
                sent1_ids = self.tokenizer.convert_tokens_to_ids(sent1_tokens)

                sent2_tokens = [self.bos_token] + self.tokenizer.tokenize(sent2) + [self.eos_token]
                sent2_ids = self.tokenizer.convert_tokens_to_ids(sent2_tokens)

                instance.append([label, sent1_ids, sent2_ids])

                # query = " ".join(jieba.cut(current_instance[0]))
                # sentp = " ".join(jieba.cut(current_instance[1]))
                # sentn = " ".join(jieba.cut(current_instance[2]))
                #
                # query_tokens = [self.bos_token] + self.tokenizer.tokenize(query) + [self.eos_token]
                # query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
                #
                # sentp_tokens = [self.bos_token] + self.tokenizer.tokenize(sentp) + [self.eos_token]
                # sentp_ids = self.tokenizer.convert_tokens_to_ids(sentp_tokens)
                #
                # sentn_tokens = [self.bos_token] + self.tokenizer.tokenize(sentn) + [self.eos_token]
                # sentn_ids = self.tokenizer.convert_tokens_to_ids(sentn_tokens)
                #
                # instance.append([query_ids, sentp_ids, sentn_ids])

                if len(instance) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(instance) <= 0:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        return instance
