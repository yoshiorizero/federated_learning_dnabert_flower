from torch.utils.data import DataLoader
import glob
import logging
import os
import pickle
import random
import re
import shutil
import torchvision.datasets
import torch
import flwr as fl
import argparse
from copy import deepcopy
from multiprocessing import Pool
from collections import OrderedDict
import warnings
from typing import Dict, List, Tuple
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from optimization import AdamW, get_linear_schedule_with_warmup
from configuration_bert import BertConfig
from modeling_bert import BertForMaskedLM
from tokenization_bert import BertTokenizer
from tokenization_dna import DNATokenizer
from tokenization_utils import PreTrainedTokenizer
from modeling_utils import PreTrainedModel
from run_pretrain import LineByLineTextDataset, TextDataset

# batch_size = 16
# epochs = 5
# val_steps = 32

# config_path = "./config.json"

warnings.filterwarnings("ignore")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}

MODEL_CLASSES = {
    "dna": (BertConfig, BertForMaskedLM, DNATokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
}

def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    
    mask_list = MASK_LIST[tokenizer.kmer]

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
        mask_centers = set(torch.where(masked_index==1)[0].tolist())
        new_centers = deepcopy(mask_centers)
        for center in mask_centers:
            for mask_number in mask_list:
                current_index = center + mask_number
                if current_index <= end and current_index >= 1:
                    new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[i][new_centers] = True
    

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


class MyClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        tokenizer: PreTrainedTokenizer,
        args,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.tokenizer = tokenizer
        self.args = args

    def set_parameters(self, parameters: List[np.ndarray]):
        """Loads a model and replaces it parameters with the ones given."""
        args = self.args
        config_class, model_class, _ = MODEL_CLASSES[args.model_type]

        if args.config_name:
            model_config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            model_config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            model_config = config_class()

        # text = "C G A T A T A G"
        # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))

        if args.block_size <= 0:
            args.block_size = self.tokenizer.max_len
            # Our input block size will be the max possible for the model
        else:
            args.block_size = min(args.block_size, self.tokenizer.max_len)

        if args.model_name_or_path:
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=model_config,
                cache_dir=args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = model_class(config=model_config)

        # model.to(args.device)
        print("Getting params...")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        # model.load_state_dict(state_dict, strict=True)

        return model

        # model = Model(config_path)
        # params_dict = zip(model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # model.load_state_dict(state_dict, strict=False)
        # return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        args = self.args

        model = self.set_parameters(parameters)

        model.to(device)

        print("Training Started...")

        # Get hyperparameters for this round
        # batch_size: int = config["batch_size"]
        # epochs: int = config["local_epochs"]

        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        train_sampler = RandomSampler(self.trainset) if args.local_rank == -1 else DistributedSampler(self.trainset)
        train_dataloader = DataLoader(
            self.trainset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
        )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.trainset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if args.model_name_or_path and os.path.exists(args.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                # logger.info("  Starting fine-tuning.")
                logger.info("  Starting training.")

        tr_loss, logging_loss = 0.0, 0.0

        model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_resize.resize_token_embeddings(len(self.tokenizer))

        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
        )
        set_seed(args)  # Added here for reproducibility
        ids_set = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs, labels = mask_tokens(batch, self.tokenizer, args) if args.mlm else (batch, batch)
                # print(inputs.shape)
                # print(inputs)
                # for i in range(len(inputs)):
                #     for j in range(len(inputs[i])):
                #         ids_set[str(int(inputs[i][j]))] += 1
                # print(ids_set)
                inputs = inputs.to(device)
                labels = labels.to(device)
                model.train()
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

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
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if (
                            args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate(args, model, self.tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss


                    # saving checkpoint
                    # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    #     checkpoint_prefix = "checkpoint"
                    #     # Save model checkpoint
                    #     output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    #     os.makedirs(output_dir, exist_ok=True)
                    #     model_to_save = (
                    #         model.module if hasattr(model, "module") else model
                    #     )  # Take care of distributed/parallel training
                    #     model_to_save.save_pretrained(output_dir)
                    #     tokenizer.save_pretrained(output_dir)

                    #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    #     logger.info("Saving model checkpoint to %s", output_dir)

                    #     _rotate_checkpoints(args, checkpoint_prefix)

                    #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    #     logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        parameters_prime = get_model_params(model)

        results = {
            "global_step": global_step,
            "tr_loss": tr_loss,
            "tr_loss / global_setp": tr_loss / global_step
        }

        num_examples_train = len(self.trainset)

        print("Training over.")

        # return global_step, tr_loss / global_step
        return parameters_prime, num_examples_train, results

        # results = utils.train(model, trainLoader, valLoader, epochs, self.device)

        # parameters_prime = utils.get_model_params(model)
        # num_examples_train = len(trainset)

        # return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        prefix=""
        args = self.args

        eval_output_dir = args.output_dir

        model = self.set_parameters(parameters)

        if args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly

        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(self.testset)
        eval_dataloader = DataLoader(
            self.testset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
        )

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(self.testset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = mask_tokens(batch, self.tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write(str(float(perplexity)) + "\n")
                # writer.write("%s = %s\n" % (key, str(result[key])))

        return result


def client_dry_run(device: str = "cpu"):
    """Weak tests to check whether all client methods are working as
    expected."""

    # model = utils.load_efficientnet(classes=10)
    # trainset, testset = utils.load_partition(0)
    # trainset = torch.utils.data.Subset(trainset, range(10))
    # testset = torch.utils.data.Subset(testset, range(10))
    # client = MyClient(trainset, testset, device)
    # client.fit(
    #     utils.get_model_params(model),
    #     {"batch_size": 4, "local_epochs": 1},
    # )

    # client.evaluate(utils.get_model_params(model), {"val_steps": 32})

    print("Dry Run Successful")


def main() -> None:
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        # "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
        "--train_data_file", default="6_3k.txt", type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,

        default="./fd_out",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", default="dna", type=str, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default="6_3k.txt",
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="./output6/base",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        # 0.15
        "--mlm_probability", type=float, default=0.025, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default="config.json",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="dna6",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        # default=-1,
        default=512,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    # 4
    parser.add_argument("--per_gpu_train_batch_size", default=10, type=int, help="Batch size per GPU/CPU for training.")
    
    # 6
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    # 1
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=25,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # 5e-5
    parser.add_argument("--learning_rate", default=4e-4, type=float, help="The initial learning rate for Adam.")
    # 0.0
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    # 1e-8
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    # 0.999
    parser.add_argument("--beta2", default=0.98, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # 1.0
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    # 1
    parser.add_argument("--n_process", type=int, default=24, help="")

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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--device", type=str, default="", help="For device.")
    parser.add_argument("--n_gpu", type=int, default=1, help="For gpus.")
    
    # Flower
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of DATASET to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # print(device)

    # Setup logging
    logging.basicConfig(
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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab


    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    args = parser.parse_args()

    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    # )

    _, _, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.dry:
        client_dry_run(device)
    else:
        # Load a subset of DATASET to simulate the local data partition

        if args.do_train:
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
                
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

            if args.local_rank == 0:
                torch.distributed.barrier()

        # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        # Start Flower client
        
        client = MyClient(train_dataset, eval_dataset, device, tokenizer, args)
        print("Start client...")

        # fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
        fl.client.start_numpy_client(
                server_address='localhost:5002',
                client=client,
                grpc_max_message_length = 1024*1024*1024
                )


if __name__ == "__main__":
    main()
