#! /usr/bin/python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.optim import AdamW
from transformers import (BertConfig, BertForTokenClassification,
                          BertTokenizer, LayoutLMConfig,
                          LayoutLMForTokenClassification, RobertaConfig,
                          RobertaForTokenClassification, RobertaTokenizer,
                          LayoutLMv3Config, LayoutLMv3ForTokenClassification,
                          LayoutLMv3Processor,
                          get_linear_schedule_with_warmup)
from utils import SROIEDataset, SROIEDatasetV3, evaluate, evaluate_v3, collate_fn_v3

logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss: 모델이 맞추기 어려워하는 샘플에 가중치를 더 주는 방식
    Hard sample에 집중하여 클래스 불균형 문제 해결
    """
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight
        
    def forward(self, inputs, targets):
        # inputs: (batch_size * seq_len, num_classes)
        # targets: (batch_size * seq_len,)
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight,
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # ignore_index인 부분 제외하고 평균
        mask = targets != self.ignore_index
        return focal_loss[mask].mean() if mask.sum() > 0 else focal_loss.mean()


def get_loss_fct(args, num_labels, device):
    """
    Loss 함수 생성
    - ce: 기본 CrossEntropyLoss
    - weighted: 클래스별 가중치 적용된 CrossEntropyLoss
    - focal: Focal Loss
    """
    # 클래스별 가중치 (labels.txt 순서: S-COMPANY, S-DATE, S-ADDRESS, S-TOTAL, O)
    # LayoutLMv3는 더 보수적인 가중치 사용 (과도한 가중치는 편향 유발)
    if args.model_type == "layoutlmv3":
        # v3용 완화된 가중치: 과도한 클래스 편향 방지
        class_weights = torch.tensor([
            2.0,   # S-COMPANY
            3.0,   # S-DATE
            1.5,   # S-ADDRESS
            5.0,   # S-TOTAL (완화: 20.0 → 5.0)
            0.5    # O
        ]).to(device)
        logger.info("Using relaxed class weights for LayoutLMv3")
    else:
        # v1용 기존 가중치
        class_weights = torch.tensor([
            5.0,   # S-COMPANY
            15.0,  # S-DATE
            2.0,   # S-ADDRESS
            20.0,  # S-TOTAL
            0.5    # O
        ]).to(device)
    
    if args.loss_type == "ce":
        logger.info("Using CrossEntropyLoss (default)")
        return CrossEntropyLoss(ignore_index=-100)
    elif args.loss_type == "weighted":
        logger.info(f"Using Weighted CrossEntropyLoss with weights: {class_weights.tolist()}")
        return CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    elif args.loss_type == "focal":
        logger.info(f"Using FocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
        if args.use_class_weights:
            return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, 
                           ignore_index=-100, weight=class_weights)
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=-100)
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")




MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutLMConfig, LayoutLMForTokenClassification, BertTokenizer),
    "layoutlmv3": (LayoutLMv3Config, LayoutLMv3ForTokenClassification, LayoutLMv3Processor),
}

# NOTE: DO NOT MODIFY THE FOLLOWING PATHS
# ---------------------------------------
data_dir = os.environ.get("SM_CHANNEL_TRAIN", "../input/data")
model_dir = os.environ.get("SM_MODEL_DIR", "./model")
# ---------------------------------------


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(data):
    batch = [i for i in zip(*data)]
    for i in range(len(batch)):
        if i < len(batch) - 2:
            batch[i] = torch.stack(batch[i], 0)
    return tuple(batch)


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def train(  # noqa C901
    args, train_dataset, model, tokenizer, labels, pad_token_label_id
):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=None,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Get custom loss function if needed
    custom_loss_fct = None
    if args.loss_type != "ce":
        custom_loss_fct = get_loss_fct(args, len(labels), args.device)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Loss type = %s", args.loss_type)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            
            labels_batch = batch[3].to(args.device)
            
            # For custom loss, don't pass labels to model (to get logits only)
            if custom_loss_fct is not None:
                inputs = {
                    "input_ids": batch[0].to(args.device),
                    "attention_mask": batch[1].to(args.device),
                    # Don't pass labels - we'll compute loss ourselves
                }
            else:
                inputs = {
                    "input_ids": batch[0].to(args.device),
                    "attention_mask": batch[1].to(args.device),
                    "labels": labels_batch,
                }
            
            if args.model_type in ["layoutlm"]:
                inputs["bbox"] = batch[4].to(args.device)
            inputs["token_type_ids"] = (
                batch[2].to(args.device)
                if args.model_type in ["bert", "layoutlm"]
                else None
            )  # RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            
            # Compute loss
            if custom_loss_fct is not None:
                # outputs[0] is logits when labels not provided
                logits = outputs[0]  # (batch_size, seq_len, num_labels)
                # Flatten for loss computation
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels_batch.view(-1)
                loss = custom_loss_fct(logits_flat, labels_flat)
            else:
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank in [-1, 0] and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(
                            args,
                            model,
                            tokenizer,
                            labels,
                            pad_token_label_id,
                            mode="dev",
                        )

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    model_dir = os.path.join(
                        args.model_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(model_dir)
                    tokenizer.save_pretrained(model_dir)
                    torch.save(args, os.path.join(model_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", model_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def train_v3(args, train_dataset, model, processor, labels):
    """LayoutLMv3용 train 함수 - 이미지 포함"""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn_v3,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Get custom loss function if needed
    custom_loss_fct = None
    if args.loss_type != "ce":
        custom_loss_fct = get_loss_fct(args, len(labels), args.device)

    # Train!
    logger.info("***** Running LayoutLMv3 training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Loss type = %s", args.loss_type)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "bbox": batch["bbox"].to(args.device),
                "pixel_values": batch["pixel_values"].to(args.device),
            }
            labels_batch = batch["labels"].to(args.device)
            
            if custom_loss_fct is not None:
                # Custom loss: don't pass labels
                outputs = model(**inputs)
                logits = outputs.logits
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels_batch.view(-1)
                loss = custom_loss_fct(logits_flat, labels_flat)
            else:
                inputs["labels"] = labels_batch
                outputs = model(**inputs)
                loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def main():  # noqa C901
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )

    ## Other parameters
    parser.add_argument(
        "--data_dir",
        default=data_dir,
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--labels",
        default=os.path.join(data_dir, "labels.txt"),
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--model_dir",
        default=model_dir,
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_model_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    
    # Loss function arguments
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "weighted", "focal"],
        help="Loss function type: ce (CrossEntropy), weighted (Weighted CE), focal (Focal Loss)",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for Focal Loss",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss (higher = more focus on hard samples)",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights with Focal Loss",
    )
    parser.add_argument(
        "--freeze_visual_encoder",
        action="store_true",
        help="Freeze visual encoder (patch_embed, pos_embed) for LayoutLMv3. Useful with small datasets.",
    )

    args = parser.parse_args()

    if os.path.exists(args.model_dir) and os.listdir(args.model_dir) and args.do_train:
        if not args.overwrite_model_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_model_dir to overcome.".format(
                    args.model_dir
                )
            )
        else:
            if args.local_rank in [-1, 0]:
                shutil.rmtree(args.model_dir, ignore_errors=True)

    if not os.path.exists(args.model_dir) and args.do_eval:
        raise ValueError(
            "Output directory ({}) does not exist. Please train and save the model before inference stage.".format(
                args.model_dir
            )
        )

    if (
        not os.path.exists(args.model_dir)
        and args.do_train
        and args.local_rank in [-1, 0]
    ):
        os.makedirs(args.model_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.model_dir, "train.log")
        if args.local_rank in [-1, 0]
        else None,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # LayoutLMv3는 Processor 사용, 나머지는 Tokenizer 사용
    if args.model_type == "layoutlmv3":
        processor = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            apply_ocr=False,  # OCR 결과 이미 있음
        )
        logger.info("LayoutLMv3Processor will handle image preprocessing (224x224)")
        tokenizer = processor.tokenizer  # 내부 tokenizer 참조
    else:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        processor = None
    
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # LayoutLMv3 visual encoder freeze (적은 데이터셋에서 효과적)
    if args.model_type == "layoutlmv3" and args.freeze_visual_encoder:
        frozen_params = 0
        for name, param in model.named_parameters():
            # patch_embed만 freeze (이미지 패치 임베딩)
            if 'patch_embed' in name:
                param.requires_grad = False
                frozen_params += param.numel()
        logger.info(f"Frozen visual encoder parameters: {frozen_params:,}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.model_type == "layoutlmv3":
            train_dataset = SROIEDatasetV3(args, processor, labels, mode="train")
            global_step, tr_loss = train_v3(
                args, train_dataset, model, processor, labels
            )
        else:
            train_dataset = SROIEDataset(
                args, tokenizer, labels, pad_token_label_id, mode="train"
            )
            global_step, tr_loss = train(
                args, train_dataset, model, tokenizer, labels, pad_token_label_id
            )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.model_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.model_dir)

        logger.info("Saving model checkpoint to %s", args.model_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.model_dir)
        
        # LayoutLMv3는 processor 저장, 나머지는 tokenizer 저장
        if args.model_type == "layoutlmv3":
            processor.save_pretrained(args.model_dir)
        else:
            tokenizer.save_pretrained(args.model_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.model_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
