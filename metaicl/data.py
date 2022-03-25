# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import random
import string
import numpy as np
import pickle as pkl
import math
import torch
import time

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import wandb
from tqdm import tqdm

class MetaICLData(object):

    def __init__(self, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1,
                 debug_data_order=False, shuffle=True, repeat_batch=1, random_seed=0):

        self.logger = logger
        self.tokenizer = tokenizer
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank
        self.debug_data_order = debug_data_order
        self.shuffle = shuffle
        self.repeat_batch = repeat_batch

        self.tensorized_inputs = None
        self.metadata = None

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size, is_training, val_split=None):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        self.logger.info(f"inputs['input_ids'].shape {shape}")
        for v in inputs.values():
            assert v.shape==shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

        if is_training and val_split is not None:
            # We will return two dataloaders, one for train and one for val
            val_size = int(val_split * len(dataset))
            self.logger.info(f"val_split {val_split}")
            self.logger.info(f"len(dataset) {len(dataset)}")
            self.logger.info(f"val_size {val_size}")
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=self.shuffle)
            validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            self.logger.info(f"len(train_loader) {len(train_loader)}")
            self.logger.info(f"len(validation_loader) {len(validation_loader)}")
            return train_loader, validation_loader
        elif is_training and val_split is None:
            sampler=RandomSampler(dataset)
        else: # Don't shuffle during test time
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        self.logger.info(f"len(dataloader) {len(dataloader)}")
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification):
        assert len(predictions)==len(self.metadata)
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not is_classification:
            return np.mean(accs)

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))

        return np.mean(f1s)

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False, for_demonstrations=False,
                               add_newlines=True):
        dp = dp.copy()
        if self.debug_data_order:
            dp['input'] = f"::TASK{dp['task']}\n" + dp['input'] # Prepend task name
        if add_newlines:
            if self.method=="direct":
                if not is_first:
                    dp["input"] = "\n\n\n" + dp["input"]
                dp["output"] = "\n" + dp["output"]
                if "options" in dp:
                    dp["options"] = ["\n" + opt for opt in dp["options"]]
            elif self.method=="channel":
                if not is_first:
                    dp["output"] = "\n\n\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n\n\n" + opt for opt in dp["options"]]
                dp["input"] = "\n" + dp["input"]
            else:
                raise NotImplementedError()
        else:
            if not is_first:
                if self.method=="direct":
                    dp["input"] = " " + dp["input"]
                elif self.method=="channel":
                    dp["output"] = " " + dp["output"]
                    if "options" in dp:
                        dp["options"] = [" "+opt for opt in dp["options"]]
                else:
                    raise NotImplementedError()
            if self.method=="direct":
                dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" " + opt for opt in dp["options"]]
            elif self.method=="channel":
                dp["input"] = " " + dp["input"]
            else:
                raise NotImplementedError()

        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"])["input_ids"]

            if "task" in dp:
                if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                        len(input_tokens)+len(output_tokens)+2>self.max_length_per_example:
                    input_tokens = input_tokens[:self.max_length_per_example // 2]
                    output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

                elif len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                    if dp["task"].startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                    else:
                        input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            assert len(dp["options"])>=2, dp
            assert dp["output"] in dp["options"]
            option_tokens = [self.tokenizer(option)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            option_tokens = [dp["options"].index(dp["output"])]

            if self.method=="direct":
                return input_tokens, output_tokens, option_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens, option_tokens
            else:
                raise NotImplementedError()

    def _tensorize_for_training(self, train_data):
        if len(train_data) < 5:
            return None
        # Train data is a flat list of [json_obj, json_obj, json_obj, ...] where each json_obj is an example from relevant train.jsonl files
        try:
            for dp in train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)

            # each datapoint: passage, question, options, output
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id

            input_ids, attention_mask, token_type_ids = [], [], []
            n_answers = []

            # few-shot learning
            if self.use_demonstrations:
                k = min(len(train_data) - 1, self.k)

                # Apply tokenization to all datapoints, so we can conveniently grab the ones we need later
                first_tokenized = []
                nonfirst_tokenized = []
                # first/nonfirst simply differs based on padding nonfirst with \n\n\n
                # so that text isn't joined up on concatenating
                for dp in train_data:
                    # _prepro_each_datapoint returns
                    # (Direct) [input_tokens, output_tokens]
                    # (Channel) [output_tokens, input_tokens]
                    first_tokenized.append(self._prepro_each_datapoint(
                        dp, is_first=True, is_training=True))
                    nonfirst_tokenized.append(self._prepro_each_datapoint(
                        dp, is_first=False, is_training=True))

                N=1

                def _draw_random(tot, n, exclude_indices):
                    """
                    This is equivalent to 

                    candidate_idxs = list(range(num_examples)).remove(exclude_idx)
                    return np.random.choice(candidate_idxs, size=n, replace=False)
                    """
                    r = np.random.choice([i for i in range(tot) if i not in exclude_indices])
                    if n==1:
                        return [r]
                    return [r] + _draw_random(tot, n-1, exclude_indices | set([r]))
                
                def _draw_sequential(tot, n, curr_idx):
                    # We will draw the n previous examples (up to but excluding curr_idx)
                    idxs = list(range(tot))
                    if curr_idx - n >= 0:
                        return idxs[curr_idx-n:curr_idx]
                    else:
                        # (loop around the end of the list if exclude_indices < n)
                        return idxs[curr_idx-n:] + idxs[:curr_idx]

                # We create a few-shot prompt for every single dp in the train data
                if self.shuffle:
                    target_order = list(range(len(train_data)))
                    random.shuffle(target_order)
                else:
                    target_order = []
                    target_idx = 0
                    for _ in train_data:
                        target_order.append(target_idx)
                        target_idx += (k + 1)
                        target_idx = target_idx % len(train_data)
                if len(set(target_order)) != len(train_data):
                    print(f"WARNING: chosen targets contain repetitions ({len(set(target_order))} unique values; expected {len(train_data)})")
                for dp_idx in target_order:
                    
                    for _ in range(N):
                        if self.shuffle:
                            # Draw k examples (demos) from the train data, without replacement, excluding the query example (dp_idx)
                            demon_indices = _draw_random(len(train_data), k, set([dp_idx]))
                        else:
                            # Draw sequentially
                            demon_indices = _draw_sequential(len(train_data), k, dp_idx)
                        
                        k_plus_one_idxs = demon_indices + [dp_idx]
                        for _ in range(self.repeat_batch):
                            # Unpack after shuffling
                            demon_indices, dp_idx = k_plus_one_idxs[:-1], k_plus_one_idxs[-1]

                            inputs = []
                            for demon_idx, index in enumerate(demon_indices):
                                if demon_idx==0:
                                    inputs += first_tokenized[index][0] + first_tokenized[index][1]
                                else:
                                    inputs += nonfirst_tokenized[index][0] + nonfirst_tokenized[index][1]
                                assert index!=dp_idx
                            # nonfirst_tokenized is a list of [input, output] tuples
                            inputs += nonfirst_tokenized[dp_idx][0]
                            outputs = nonfirst_tokenized[dp_idx][1]

                            # Go from tokenized lists of inputs and outputs to
                            # input_ids, attention_mask, token_type_ids = encoded
                            # input_ids: [*input_token_ids, *output_token_ids, padding] with len(input_ids) == max_length
                            # attention_mask: [1*len(input_token_ids), 0*len(output_token_ids), padding] with len(attention_mask) == max_length
                            # token_type_ids: [0*len(input_token_ids), 1*len(output_token_ids), padding] with len(token_type_ids) == max_length
                            encoded = prepro_sentence_pair_single(
                                inputs, outputs, self.max_length, bos_token_id, eos_token_id,
                                allow_truncation=True)

                            input_ids.append(encoded[0])
                            attention_mask.append(encoded[1])
                            token_type_ids.append(encoded[2])

                            # Shuffle
                            random.shuffle(k_plus_one_idxs)

            # zero-shot learning
            else:
                for dp in train_data:
                    inputs, outputs = self._prepro_each_datapoint(
                        dp, is_first=True, is_training=True)

                    encoded = prepro_sentence_pair_single(
                        inputs, outputs, self.max_length, bos_token_id, eos_token_id)

                    input_ids.append(encoded[0])
                    attention_mask.append(encoded[1])
                    token_type_ids.append(encoded[2])

            return dict(input_ids=torch.LongTensor(input_ids),
                        attention_mask=torch.LongTensor(attention_mask),
                        token_type_ids=torch.LongTensor(token_type_ids))
        except AssertionError as e:
            self.logger.info(("Assertion failed! Skipping", dp.get("task", None), e))
            return None


    def tensorize(self, _train_data, _test_data, options=None,
                  add_newlines=True):

        if options is not None:
            assert np.all([dp["output"] in options for dp in _train_data])
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data = [], []
        if self.use_demonstrations:
            for dp in _train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                train_data.append(dp.copy())
        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        if self.use_demonstrations:
            assert len(train_data)==self.k
            demonstrations = []
            for i, dp in enumerate(train_data):
                input_, output_ = self._prepro_each_datapoint(
                    dp, is_first=i==0, for_demonstrations=True,
                    add_newlines=add_newlines)
                demonstrations += input_ + output_

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            indices = [[i] for i in range(len(input_ids), len(input_ids)+len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_

                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, bos_token_id, eos_token_id,
                    allow_truncation=self.use_demonstrations)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def tensorize_for_training(self, train_data, keyword, seed, debug_order=False):
        # assert self.tensorize_dir is not None

        # if not os.path.exists(self.tensorize_dir):
        #     os.makedirs(self.tensorize_dir)

        # method_name = self.method + "-demon" if self.use_demonstrations else self.method
        # k_name = "%d-%d" % (len(train_data), self.k) if self.use_demonstrations else len(train_data)
        # length_name = "%d-%d" % (self.max_length, self.max_length_per_example) if self.use_demonstrations else self.max_length

        # tensorize_path = os.path.join(self.tensorize_dir,
        #                               "{}_{}_k={}_seed={}_length={}-rank=%d.pkl".format(
        #                                   keyword, method_name, k_name, seed, length_name))

        # if self.local_rank==-1:
        #     self.logger.info(tensorize_path)
        # else:
        #     self.logger.info(tensorize_path % self.local_rank)
        # all_tensorize_paths = [tensorize_path % i for i in range(self.n_gpu)]

        # if not self.do_tensorize:
        #     if not np.all([os.path.exists(_path) for _path in all_tensorize_paths]):
        #         self.logger.info("Tensorization was not done. Run with `--do_tensorize` without distributed mode"
        #                     "and then run training command again")
        #         raise NotImplementedError()

        #     if self.local_rank==-1:
        #         inputs = defaultdict(list)
        #         for i in range(self.n_gpu):
        #             with open(tensorize_path % i, "rb") as f:
        #                 curr_inputs = pkl.load(f)
        #             for k, v in curr_inputs.items():
        #                 inputs[k] += v
        #     else:
        #         assert 0<=self.local_rank<self.n_gpu
        #         with open(tensorize_path % self.local_rank, "rb") as f:
        #             inputs = pkl.load(f)

        #     self.tensorized_inputs = inputs
        #     return

        # assert self.local_rank==-1
        # # if any([os.path.exists(_path) for _path in all_tensorize_paths]):
        # #     self.logger.info("tensorize file already exists...")
        # #     return
        # for _path in all_tensorize_paths:
        #     if os.path.exists(_path):
        #         self.logger.info("Tensorize file already exists! Deleting and re-processing...")
        #         os.remove(_path)

        unique_task_names = list(dict.fromkeys([dp["task"] for dp in train_data])) # Equivalent to set(ls) but maintains order
        sharded_inputs = []
        self.logger.info("sharding inputs...")
        if self.use_demonstrations or (len(unique_task_names)>200 and len(train_data)>=1638400):
            """
            Split data into `n_task` shards
            """
            print(f"Splitting data into `n_task` ({len(unique_task_names)}) shards")
            tot = 0
            for i, curr_train_task in enumerate(tqdm(unique_task_names)):
                curr_train_data = [dp for dp in train_data if dp["task"]==curr_train_task]
                tot += len(curr_train_data)
                if self.use_demonstrations and len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 10% of the data
                    self.logger.info("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//10)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//10]
                    curr_train_data = [curr_train_data[i] for i in indices]
                elif len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 50% of the data
                    self.logger.info("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//2)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//2]
                    curr_train_data = [curr_train_data[i] for i in indices]
                sharded_inputs.append(curr_train_data)
            assert len(train_data)==tot
            assert len(sharded_inputs) == len(unique_task_names)
        else:
            """
            Simply split the data "as is" into `n_process` shards
            """
            print(f"Splitting data into n_process ({self.n_process}) shards")
            n_per_shard = math.ceil(len(train_data) / self.n_process)
            for i in range(self.n_process):
                sharded_inputs.append(train_data[i*n_per_shard:(i+1)*n_per_shard])
            assert len(sharded_inputs) == self.n_process

        inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        self.logger.info(f"len(sharded_inputs) {len(sharded_inputs)}")
        self.logger.info(f"running on {self.n_process} process(es)")
        if self.n_process==1 or not self.shuffle:
            self.logger.info("Performing single-process tensorization...")

            # Tensorize data one task at a time
            assert len(sharded_inputs) == len(unique_task_names)
            data_by_task = []
            start_t = time.time()
            for in_ in tqdm(sharded_inputs):
                self.logger.info(f"Tensorizing {in_[0]['task']}")
                out = self._tensorize_for_training(in_)
                # out is a list of (k-shot context & target, tokenized) for each M elements in the data
                # represented as a dict containing keys ["input_ids", "attention_mask", "token_type_ids"]
                # where each dict item is a M-element list of 1024-length token arrays
                if out is not None:
                    data_by_task.append({
                        'input_ids': out['input_ids'].numpy().tolist(),
                        'attention_mask': out['attention_mask'].numpy().tolist(),
                        'token_type_ids': out['token_type_ids'].numpy().tolist(),
                    })
            duration = time.time() - start_t
            self.logger.info(f"Tensorizing {len(train_data)} datapoints took {duration}s ({duration / len(train_data)}s per datapoint)")
        
            # Go from data_by_task to a flat list of data points round-robin style
            # (task 1 example 1, task 2 example 1, ..., task 1 example 2, task 2 example 2, ...)
            still_have_data = True
            data_idx = 0
            while still_have_data:
                still_have_data = False
                for task_data in data_by_task:
                    for repeat_idx in range(self.repeat_batch): # Add repeat batches together (no round-robin)
                        idx = data_idx + repeat_idx
                        if idx >= len(task_data['input_ids']):
                            continue
                        still_have_data = True
                        for key in ["input_ids", "attention_mask", "token_type_ids"]:
                            # This appends a single datapoint (1024-token array)
                            tokenized_dp = task_data[key][idx]
                            inputs[key].append(tokenized_dp)
                data_idx += self.repeat_batch # Skip to the next batch
        else:
            print("Performing multi-process tensorization...")
            with Pool(self.n_process) as p:
                for out in p.imap_unordered(self._tensorize_for_training, sharded_inputs):
                    if out is not None:
                        for key in ["input_ids", "attention_mask", "token_type_ids"]:
                            inputs[key] += out[key].numpy().tolist()

        N = len(inputs["input_ids"]) # Number of datapoints across all tasks
        print("len(inputs['input_ids'])", len(inputs["input_ids"]))
        if self.shuffle:
            # This shuffling is for multi-gpu setups, to ensure each gpu gets a random sample of tasks
            indices = np.random.permutation(range(N))
            for k, v in inputs.items():
                inputs[k] = np.array(v)[indices]
            n_per_shard = math.ceil(N / self.n_gpu)
            for k, v in inputs.items():
                inputs[k] = v.tolist()
        # else:
        #     # Simply convert to np array
        #     for k, v in inputs.items():
        #         print(k, len(v))
        #         inputs[k] = np.array(v)
        #     n_per_shard = N

        # for i, _path in enumerate(all_tensorize_paths):
        #     start = i*n_per_shard
        #     end = (i+1)*n_per_shard
        #     curr_inputs = {k:v[start:end].tolist() for k, v in inputs.items()}
        #     with open(_path, "wb") as f:
        #         pkl.dump(curr_inputs, f)
        #     self.logger.info("Preprocessing done for i=%d" % i)

        # self.logger.info("Finish saving preprocessed data ...")

        self.tensorized_inputs = inputs

    def print_batch(self, batch, batch_idx=None):
        input_ids = batch[0][0]
        attention_mask = batch[1][0]
        token_type_ids = batch[2][0]
        
        if batch_idx is not None:
            self.logger.info(f"BATCH {batch_idx}")
        # self.logger.info(f"\n\n\n\n\n\n EXAMPLE ------------------------------------------")
        # text = "Checking the first example..."
        # input_ids = self.tensorized_inputs["input_ids"][idx]
        # token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        self.logger.info(f'\ninput_ids: ({len(input_ids)}) {input_ids}')
        self.logger.info(f'\nattention_mask: ({len(attention_mask)}) {attention_mask}')
        self.logger.info(f'\ntoken_type_ids: ({len(token_type_ids)}) {token_type_ids}')
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(attention_mask)!=list:
            attention_mask = attention_mask.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        # Input is all elements up to the first occurence of '1' in token_type_ids
        input_ids_in = input_ids[:token_type_ids.index(1)]
        self.logger.info(f'\n\ncontext_input_ids: ({len(input_ids_in)}) {input_ids_in}')
        # Output is all elements corresponding to '1' in token_type_ids
        input_ids_out = [_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1]
        self.logger.info(f'\nanswer_input_ids: ({len(input_ids_out)}) {input_ids_out}')
        self.logger.info(f'\nTotal ids excluding padding: {len(input_ids_in) + len(input_ids_out)} (if ==1024, truncation probably occurred)\n')

        self.logger.info("\n\nCONTEXT:\n")
        context_text = self.tokenizer.decode(input_ids_in)
        self.logger.info(context_text)
        self.logger.info("\n\nANSWER:\n")
        answer_text = self.tokenizer.decode(input_ids_out)
        self.logger.info(answer_text)

        # if self.local_rank<=0:
        #     self.logger.info(text)
        text = context_text + answer_text
        return text

    def print_tensorized_example(self, return_string=False, n_examples=10):
        assert self.tensorized_inputs is not None

        idx = 0
        for idx in range(n_examples):
            text = f"\n\n\n\n\n\n{idx}-TH EXAMPLE ------------------------------------------"
            # text = "Checking the first example..."
            input_ids = self.tensorized_inputs["input_ids"][idx]
            token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
            text += f'\ninput_ids: ({len(input_ids)}) {input_ids}'
            text += f'\ntoken_type_ids: ({len(token_type_ids)}) {token_type_ids}'
            if type(input_ids)!=list:
                input_ids = input_ids.numpy().tolist()
            if type(token_type_ids)!=list:
                token_type_ids = token_type_ids.numpy().tolist()

            # Input is all elements up to the first occurence of '1' in token_type_ids
            input_ids_in = input_ids[:token_type_ids.index(1)]
            text += f'\n\ncontext_input_ids: ({len(input_ids_in)}) {input_ids_in}'
            # Output is all elements corresponding to '1' in token_type_ids
            input_ids_out = [_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1]
            text += f'\nanswer_input_ids: ({len(input_ids_out)}) {input_ids_out}'
            text += f'\nTotal ids excluding padding: {len(input_ids_in) + len(input_ids_out)} (if ==1024, truncation probably occurred)\n'


            text += "\n\nCONTEXT:\n"
            text += self.tokenizer.decode(input_ids_in)
            text += "\n\nANSWER:\n"
            text += self.tokenizer.decode(input_ids_out)

            if return_string:
                return text

            if self.local_rank<=0:
                self.logger.info(text)

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id,
                                allow_truncation=False):
    """
    ids1: Tokenized input_ids for the input
    ids2: Tokenized input_ids for the output
    """

    #if bos_token_id is not None:
    #    ids1 = [bos_token_id] + ids1
    #if eos_token_id is not None:
    #    ids2 = ids2 + [eos_token_id]
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        # print("TRUNCATING inside prepro_sentence_pair_single", len(ids1) + len(ids2), max_length)
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length, (len(ids1), len(ids2), max_length)

    n_mask = max_length-len(ids1)-len(ids2) # Pad with zeros to the full max_length
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    """
    Returns:

    input_ids      = [IN0, IN1, ..., OUT0, OUT1, ..., 0, 0, 0]
    attention_mask = [1  , 1  , ..., 0   , 0   , ..., 0, 0, 0]
    token_type_ids = [0  , 0  , ..., 1   , 1   , ..., 0, 0, 0] # Indicates which tokens belong to (x, y)
    """
    return input_ids, attention_mask, token_type_ids

# THIS FUNCTION IS NEVER USED
def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}

