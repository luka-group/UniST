import argparse
import json
import os
import logging
import glob
import random
import numpy as np
import torch
import copy
from tqdm import tqdm, trange
from transformers import RobertaTokenizer, RobertaConfig, AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from eval_metric import macro, macro_fewshot, tacred_f1

from data import TACREDDataset, UFETDataset, MAVENDataset, MAVENTestDataset, FewRelDataset
from model import UniSTModel

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_datasets, model, tokenizer):
    """train the model"""     
    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=lambda x: list(map(list, zip(*x))))
    
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_steps)

    num_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility 

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            sent, pos, neg = batch[0], batch[1], batch[2]
            model.train()
            sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=args.max_sent_length, return_tensors="pt", is_split_into_words=True).to(args.device)                
            pos_inputs = tokenizer(pos, padding=True, truncation=True, max_length=args.max_label_length, return_tensors="pt").to(args.device)
            neg_inputs = tokenizer(neg, padding=True, truncation=True, max_length=args.max_label_length, return_tensors="pt").to(args.device)
            
            inputs = {
                "sent_input_ids": sent_inputs["input_ids"],
                "pos_input_ids": pos_inputs["input_ids"],
                "neg_input_ids": neg_inputs["input_ids"],
                "sent_attention_mask": sent_inputs["attention_mask"],
                "pos_attention_mask": pos_inputs["attention_mask"],
                "neg_attention_mask": neg_inputs["attention_mask"],
            }

            loss = model(**inputs)[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps               
            loss.backward()
                       
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                num_steps += 1


        if args.local_rank in [-1, 0] and args.logging_epochs > 0 and (epoch + 1) % args.logging_epochs == 0:
            if args.eval_during_training:  
                model_to_eval = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                if "tacred" in args.eval_tasks:
                    eval_tacred(args, eval_datasets["tacred_dev"], model_to_eval, tokenizer, num_epochs=epoch+1, split="dev")
                    eval_tacred(args, eval_datasets["tacred_test"], model_to_eval, tokenizer, num_epochs=epoch+1, split="test")
                if "ufet" in args.eval_tasks:
                    threshold = eval_ufet(args, eval_datasets["ufet_dev"], model_to_eval, tokenizer, num_epochs=epoch+1, split="dev")["Threshold"]
                    eval_ufet(args, eval_datasets["ufet_test"], model_to_eval, tokenizer, num_epochs=epoch+1, split="test", threshold=threshold)
                if "maven" in args.eval_tasks:
                    eval_maven(args, eval_datasets["maven_dev"], model_to_eval, tokenizer, num_epochs=epoch+1, split="dev")
                if "fewrel" in args.eval_tasks:
                    eval_fewrel(args, eval_datasets["fewrel_dev"], model_to_eval, tokenizer, num_epochs=epoch+1, split="dev")

        if args.local_rank in [-1, 0] and args.save_epochs > 0 and (epoch+1) % args.save_epochs == 0 and epoch > 0.7*args.num_train_epochs:
            # Save model checkpoint                    
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(epoch+1))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                               
            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin")) 

    return 0

def eval_tacred(args, eval_dataset, model, tokenizer, num_epochs="", split=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: list(map(list, zip(*x))))

    logger.info("***** Running tacred %s evaluation at %s *****", split, num_epochs)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    dists = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        labelset = eval_dataset.labelset
        labelset_inputs = tokenizer(labelset, padding=True, truncation=True, max_length=args.max_label_length, return_tensors="pt").to(args.device)
        labelset_embeddings = model.embed(**labelset_inputs)
    
    for sent, pos, _ in tqdm(eval_dataloader, desc="Evaluating"):
        labels.extend(pos)
        sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=args.max_sent_length, return_tensors="pt", is_split_into_words=True).to(args.device)
                         
        with torch.no_grad():            
            sent_embeddings = model.embed(**sent_inputs)
            for i in range(len(sent_embeddings)):
                sent_embedding = sent_embeddings[i].expand(labelset_embeddings.shape)
                dist = model.dist_fn(sent_embedding, labelset_embeddings).detach().cpu().numpy().tolist()
                dists.append(dist)  

    preds = []
    for i, dist in enumerate(dists):
        pred = min(labelset, key=lambda x:dist[labelset.index(x)])
        preds.append(pred)

    assert len(preds) == len(eval_dataset)
    p, r, f1 = tacred_f1(labels, preds)

    results = {
        "Micro precision": p,
        "Micro recall": r,
        "Micro f1": f1
    }
    
    for key in results:
        logger.info("  %s = %s", key, str(results[key]))

    return results


def eval_ufet(args, eval_dataset, model, tokenizer, num_epochs="", split="", threshold=None, threshold_step=1e-2):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: list(map(list, zip(*x))))

    logger.info("***** Running ufet %s evaluation at %s *****", split, num_epochs)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Threshold step = %f", threshold_step)


    dists = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        labelset = eval_dataset.labelset
        labelset_inputs = tokenizer(labelset, padding=True, truncation=True, max_length=args.max_label_length, return_tensors="pt").to(args.device)
        labelset_embeddings = model.embed(**labelset_inputs)
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        sent, pos_all = batch[0], batch[3]
        labels.extend(pos_all)
        sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=args.max_sent_length, return_tensors="pt", is_split_into_words=True).to(args.device)

        with torch.no_grad():
            sent_embeddings = model.embed(**sent_inputs)
            for i in range(len(sent_embeddings)):
                sent_embedding = sent_embeddings[i].expand(labelset_embeddings.shape)
                dist = model.dist_fn(sent_embedding, labelset_embeddings).detach().cpu().numpy().tolist()
                dists.append(dist)
    
    if threshold is None:
        # Find the best threshold
        threshold = np.array(dists).min()

        best_f1 = 0.0
        best_f1_p = 0.0
        best_f1_r = 0.0
        best_threshold = 0.0
        while threshold <= np.array(dists).max():
            preds = []
            for dist in dists:
                pred = [labelset[i] for i, val in enumerate(dist) if val < threshold]
                preds.append(pred)

            assert len(preds) == len(eval_dataset)
            p, r, f1 = macro(labels=labels, preds=preds)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_p = p
                best_f1_r = r
                best_threshold = threshold

            threshold += threshold_step

        results = {
            "Threshold": best_threshold,
            "Macro precision": best_f1_p,
            "Macro recall": best_f1_r,
            "Macro f1": best_f1
        }

    else:
        with open("../data/ufet/test_type_frequencies_in_train_set.json", "r") as f:
            label2freq = json.loads(f.read())

        preds = []
        for dist in dists:
            pred = [labelset[i] for i, val in enumerate(dist) if val < threshold]
            preds.append(pred)

        zero_labels=[label for label, freq in label2freq.items() if freq == 0]
        print(len(zero_labels))
        _, _, f1_zero = macro_fewshot(labels=labels, preds=preds, target_labels=zero_labels)

        few_labels=[label for label, freq in label2freq.items() if 1 <= freq <= 5]
        print(len(few_labels))
        _, _, f1_few = macro_fewshot(labels=labels, preds=preds, target_labels=few_labels)

        more_labels=[label for label, freq in label2freq.items() if 6 <= freq <= 10]
        print(len(more_labels))
        _, _, f1_more = macro_fewshot(labels=labels, preds=preds, target_labels=more_labels)

        p_all, r_all, f1_all = macro(labels=labels, preds=preds)

        results = {
            "Threshold": threshold,
            "Macro p-all": p_all,
            "Macro r-all": r_all,
            "Macro f1-all": f1_all,
            "Macro f1-zero": f1_zero,
            "Macro f1-few": f1_few,
            "Macro f1-more": f1_more,
        }

    
    for key in results:
        logger.info("  %s = %s", key, str(results[key]))

    return results

def eval_maven(args, eval_dataset, model, tokenizer, num_epochs="", split=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: list(map(list, zip(*x))))

    logger.info("***** Running maven %s evaluation at %s *****", split, num_epochs)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    dists = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        labelset = eval_dataset.labelset
        labelset_inputs = tokenizer(labelset, padding=True, truncation=True, max_length=args.max_label_length, return_tensors="pt").to(args.device)
        labelset_embeddings = model.embed(**labelset_inputs)
       
    for sent, pos, _ in tqdm(eval_dataloader, desc="Evaluating"):
        labels.extend(pos)
        sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=args.max_sent_length, return_tensors="pt", is_split_into_words=True).to(args.device)
                         
        with torch.no_grad():            
            sent_embeddings = model.embed(**sent_inputs)
            for i in range(len(sent_embeddings)):
                sent_embedding = sent_embeddings[i].expand(labelset_embeddings.shape)
                dist = model.dist_fn(sent_embedding, labelset_embeddings).detach().cpu().numpy().tolist()
                dists.append(dist)  

    preds = []
    for i, dist in enumerate(dists):
        pred = min(labelset, key=lambda x:dist[labelset.index(x)])
        preds.append(pred)

    assert len(preds) == len(eval_dataset)
    results = {
        "Micro precision": precision_score(labels, preds, average="micro")
    }
    
    for key in results:
        logger.info("  %s = %s", key, str(results[key]))

    return results


def test_maven(args, eval_dataset, model, tokenizer, num_epochs="", split=""):
    """
    generate as MAVEN test format
    prediction file to submit to leaderboard: each line a json
       id: docid, predictions: list
            each ele = dict[id=(sent)id, type_id=event label (start from 0 = None)]
    """

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: list(map(list, zip(*x))))

    logger.info("***** Running maven %s evaluation at %s *****", split, num_epochs)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    docids = []
    ids = []
    dists = []
    model.eval()
    
    with torch.no_grad():
        labelset = eval_dataset.labelset
        labelset_inputs = tokenizer(labelset, padding=True, truncation=True, max_length=args.max_label_length, return_tensors="pt").to(args.device)
        labelset_embeddings = model.embed(**labelset_inputs)
       
    for sent, docid, id in tqdm(eval_dataloader, desc="Evaluating"):
        docids.extend(docid)
        ids.extend(id)
        sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=args.max_sent_length, return_tensors="pt", is_split_into_words=True).to(args.device)
                         
        with torch.no_grad():            
            sent_embeddings = model.embed(**sent_inputs)
            for i in range(len(sent_embeddings)):
                sent_embedding = sent_embeddings[i].expand(labelset_embeddings.shape)
                dist = model.dist_fn(sent_embedding, labelset_embeddings).detach().cpu().numpy().tolist()
                dists.append(dist)
        
    preds = []
    for i, dist in enumerate(dists):
        pred = min(labelset, key=lambda x:dist[labelset.index(x)])
        preds.append(pred)

    submissions = copy.deepcopy(eval_dataset.negative_trigger)
    for docid, id, pred in zip(docids, ids, preds):
        submissions[docid].append({"id": id, "type_id": labelset.index(pred)+1}) 

    # sanity check
    with open(os.path.join(args.output_dir, "%s_%s_results.jsonl" % (split, num_epochs)), "w") as f:
        for docid, preds in submissions.items():
            f.write(json.dumps({"id": docid, "predictions": preds}) + "\n")


def eval_fewrel(args, eval_dataset, model, tokenizer, num_epochs="", split=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: list(map(list, zip(*x))))

    # Eval!
    logger.info("***** Running fewrel %s evaluation at %s *****", split, num_epochs)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    dists = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        labelset = eval_dataset.labelset
        labelset_inputs = tokenizer(labelset, padding=True, truncation=True, max_length=20, return_tensors="pt").to(args.device)
        labelset_embeddings = model.embed(**labelset_inputs)

    for sent, pos, _ in tqdm(eval_dataloader, desc="Evaluating"):
        labels.extend(pos)
        sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=args.max_sent_length, return_tensors="pt", is_split_into_words=True).to(args.device)
                         
        with torch.no_grad():            
            sent_embeddings = model.embed(**sent_inputs)
            for i in range(len(sent_embeddings)):
                sent_embedding = sent_embeddings[i].expand(labelset_embeddings.shape)
                dist = model.dist_fn(sent_embedding, labelset_embeddings).detach().cpu().numpy().tolist()
                dists.append(dist)  
              
    N = [10, 5]
    score_all_run = []

    n_runs = 10
    for run in range(n_runs):
        scores = []
        for n in N:
            preds = []
            for i, label in enumerate(labels): 
                dist = dists[i]   		
                labelset_copy = labelset.copy()
                labelset_copy.remove(label)
                n_way_labelset = [label] + random.sample(labelset_copy, n-1)
                pred = min(n_way_labelset, key=lambda x:dist[labelset.index(x)])
                preds.append(pred)
            scores.append(accuracy_score(labels, preds))
        score_all_run.append(scores)
    score_all_run = list(zip(*score_all_run))
    

    results = {
        "10-way-0-shot Accuracy": sum(score_all_run[0])/len(score_all_run[0]),
        "5-way-0-shot Accuracy": sum(score_all_run[1])/len(score_all_run[1]),
    }
    
    for key in results:
        logger.info("  %s = %s", key, str(results[key]))

    return results



def main():
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument("--output_dir", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")     
    parser.add_argument("--data_dir", type=str, 
                        help="The input data directory.")
    parser.add_argument("--train_tasks", nargs="+",
                        help="Specify what task to train the model on")
    parser.add_argument("--eval_tasks", nargs="+",
                        help="Specify what task to evaluate the model on")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")   
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--eval_dir", default=None, type=str,
                        help="Where to find model checkpoints for evluation")


    # model config
    parser.add_argument("--margin", default=0.1, type=float,
                        help="The margin of the triplet loss function.")
    parser.add_argument("--no_task_desc", action="store_true",
                        help="Remove task desc.")
    parser.add_argument("--max_sent_length", default=160, type=int,
                        help="The maximum total input sentence length after tokenization. Sequences longer "
                             "than this will be truncated.")
    parser.add_argument("--max_label_length", default=20, type=int,
                        help="The maximum total input label length after tokenization. Sequences longer "
                             "than this will be truncated.")

    # procedure                             
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation on the dev set.")
    parser.add_argument("--eval_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--logging_epochs", type=int, default=10,
                        help="Log every X epochs.")
    parser.add_argument("--save_epochs", type=int, default=10,
                        help="Save checkpoint every X epochs.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints ending with step number")

    # optimizer
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")

    # gpu
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    # seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()


    if any(task for task in args.train_tasks if task not in ["tacred", "ufet", "maven", "fewrel"]):
        raise ValueError("train tasks must be tacred, ufet, maven, or fewrel")
    
    if any(task for task in args.eval_tasks if task not in ["tacred", "ufet", "maven", "fewrel"]):
        raise ValueError("eval tasks must be tacred, ufet, maven, or fewrel")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(filename=os.path.join(args.output_dir, "logs.log"),
    	                format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config = RobertaConfig.from_pretrained(args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.margin = args.margin
    
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path,
                                                cache_dir=args.cache_dir if args.cache_dir else None,
                                                add_prefix_space=True)

    model = UniSTModel.from_pretrained(args.model_name_or_path,
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)


    # add new token
    special_tokens_dict = {"additional_special_tokens":["<E>", "</E>", "<SUBJ>", "</SUBJ>", "<OBJ>", "</OBJ>", "<T>", "</T>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens")
    model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.


    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Prepare datasets
    if args.do_train or args.do_eval:
        #if args.local_rank not in [-1, 0]:
        #    torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        # tacred
        tacred_train_dataset = (
            TACREDDataset(os.path.join(args.data_dir, "tacred/train.json"), no_task_desc=args.no_task_desc)
            if "tacred" in args.train_tasks
            else None
        )
        tacred_dev_dataset = (
            TACREDDataset(os.path.join(args.data_dir, "tacred/dev.json"), no_task_desc=args.no_task_desc)
            if "tacred" in args.eval_tasks
            else None
        )
        tacred_test_dataset = (
            TACREDDataset(os.path.join(args.data_dir, "tacred/test.json"), no_task_desc=args.no_task_desc)
            if "tacred" in args.eval_tasks
            else None
        )

        # ufet
        ufet_train_dataset = (
            UFETDataset(os.path.join(args.data_dir, "ufet/train.json"), os.path.join(args.data_dir, "ufet/ufet_labels.txt"), no_duplicates=False, no_task_desc=args.no_task_desc)
            if "ufet" in args.train_tasks
            else None
        )
        if ufet_train_dataset and len(args.train_tasks) > 1:
            ufet_train_dataset.data *= 10
        ufet_dev_dataset = (
            UFETDataset(os.path.join(args.data_dir, "ufet/dev.json"), os.path.join(args.data_dir, "ufet/ufet_labels.txt"), no_duplicates=True, no_task_desc=args.no_task_desc) 
            if "ufet" in args.eval_tasks
            else None
        )
        ufet_test_dataset = (
            UFETDataset(os.path.join(args.data_dir, "ufet/test.json"), os.path.join(args.data_dir, "ufet/ufet_labels.txt"), no_duplicates=True, no_task_desc=args.no_task_desc)
            if "ufet" in args.eval_tasks
            else None
        )

        # maven
        maven_train_dataset = (
            MAVENDataset(os.path.join(args.data_dir, "maven/train.jsonl"), no_task_desc=args.no_task_desc)
            if "maven" in args.train_tasks
            else None
        )
        maven_dev_dataset = (
            MAVENDataset(os.path.join(args.data_dir, "maven/valid.jsonl"), no_task_desc=args.no_task_desc)
            if "maven" in args.eval_tasks
            else None
        )

        # fewrel
        fewrel_train_dataset = (
            FewRelDataset(os.path.join(args.data_dir, "fewrel/train_wiki_processed.jsonl"), os.path.join(args.data_dir, "fewrel/train_labels.txt"), no_task_desc=args.no_task_desc)
            if "fewrel" in args.train_tasks
            else None
        )
        fewrel_dev_dataset = (
            FewRelDataset(os.path.join(args.data_dir, "fewrel/val_wiki_processed.jsonl"), os.path.join(args.data_dir, "fewrel/val_labels.txt"), no_task_desc=args.no_task_desc)
            if "fewrel" in args.eval_tasks
            else None
        )

        train_dataset = ConcatDataset([
            dataset for dataset in [tacred_train_dataset, ufet_train_dataset, maven_train_dataset, fewrel_train_dataset]
            if dataset is not None
        ])
        eval_datasets = {
            "tacred_dev": tacred_dev_dataset,
            "tacred_test": tacred_test_dataset,
            "ufet_dev": ufet_dev_dataset,
            "ufet_test": ufet_test_dataset,
            "maven_dev": maven_dev_dataset,
            "fewrel_dev": fewrel_dev_dataset
        }

        #if args.local_rank == 0:
        #    torch.distributed.barrier()

    # Training
    if args.do_train:
        train(args, train_dataset, eval_datasets, model, tokenizer)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.eval_dir is None:
            checkpoints = [args.output_dir]
        else:
            checkpoints = [args.eval_dir]

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

        logger.info("Evaluating the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            num_epochs = checkpoint.split("-")[-1] #if len(checkpoints) > 1 else "end"    
            model = UniSTModel.from_pretrained(checkpoint)
            model.to(args.device)
            model_to_eval = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training

            if "tacred" in args.eval_tasks:
                eval_tacred(args, eval_datasets["tacred_dev"], model_to_eval, tokenizer, num_epochs=num_epochs, split="dev")
                eval_tacred(args, eval_datasets["tacred_test"], model_to_eval, tokenizer, num_epochs=num_epochs, split="test")
            if "ufet" in args.eval_tasks:
                threshold = eval_ufet(args, eval_datasets["ufet_dev"], model_to_eval, tokenizer, num_epochs=num_epochs, split="dev")["Threshold"]
                eval_ufet(args, eval_datasets["ufet_test"], model_to_eval, tokenizer, num_epochs=num_epochs, split="test", threshold=threshold)
            if "maven" in args.eval_tasks:
                eval_maven(args, eval_datasets["maven_dev"], model_to_eval, tokenizer, num_epochs=num_epochs, split="dev")
            if "fewrel" in args.eval_tasks:
                eval_fewrel(args, eval_datasets["fewrel_dev"], model_to_eval, tokenizer, num_epochs=num_epochs, split="dev")
                
    return 0


if __name__ == "__main__":
    main()