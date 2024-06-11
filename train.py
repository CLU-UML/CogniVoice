import os
import sys
import json
import torch
import wandb
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    set_seed,
    AutoTokenizer,
    TrainingArguments,
    EvalPrediction,
    default_data_collator,
    WhisperFeatureExtractor, 
    WhisperTokenizer,
    WhisperProcessor,
    AutoModelForSequenceClassification,
    Trainer
)

from cognivoice.model import Whisper, WhisperPoe
from cognivoice.data_processor import *
from cognivoice.training_args import AudioTrainingArguments, RemainArgHfArgumentParser


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = RemainArgHfArgumentParser((AudioTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        json_file=os.path.abspath(sys.argv[1])
        args, _ = parser.parse_json_file(json_file, return_remaining_args=True) #args = arg_string, return_remaining_strings=True) #parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]
    args.dataloader_num_workers = 8


    # Wandb
    args.report_to = ['wandb']
    project = 'TAUKADIAL-2024'
    group = args.task
    name = args.method
    output_dir_root = args.output_dir
    if args.use_disvoice:
        name += '_disvoice'
        output_dir_root += '_disvoice'
    if args.use_metadata:
        name += '_metadata'
        output_dir_root += '_metadata'
    if args.use_text:
        name += '_text'
        output_dir_root += '_text'
    if args.use_llama2:
        name += '_llama2'
        output_dir_root += '_llama2'
    if args.use_poe:
        name += '_poe'
        output_dir_root += '_poe'
    name += '-' + str(args.seed)

    wandb.init(project=project, group=group, name=name, config=args, id=name, resume='allow')

    set_seed(args.seed)

    # Format HF model names
    if args.method == 'wav2vec':
        args.method = 'facebook/' + args.method
    
    elif args.method.startswith('whisper'):
        args.method = 'openai/' + args.method

    from sklearn.model_selection import StratifiedKFold
    data = pd.read_csv('/data/datasets/TAUKADIAL-24/train/groundtruth.csv')
    label_col = 'dx' if args.task == 'cls' else 'mmse'
    args.metric_for_best_model = 'f1' if args.task == 'cls' else 'mse'
    args.greater_is_better = True if args.task == 'cls' else False
    kv = StratifiedKFold(n_splits=args.num_fold)

    # pred_data = TAUKADIALTestDataset(args)

    scores = []
    for fold_id, (train_idx, eval_idx) in enumerate(tqdm(kv.split(data.drop(label_col, axis=1), data[label_col]), desc='Cross Validation')):
        args.output_dir = os.path.join(output_dir_root, f'fold_{fold_id}')

        # Dataset
        train_data = TAUKADIALDataset(args, subset=train_idx)
        eval_data = TAUKADIALDataset(args, subset=eval_idx)

        # Model
        if args.method == 'wav2vec':
            model = AutoModelForSequenceClassification.from_pretrained(args.method)
        
        elif 'whisper' in args.method:

            if args.use_poe:
                model = WhisperPoe(args)
            
            else:
                model = Whisper(args)
                # model.parallelize()

        else:
            raise NotImplementedError

        
        metric = load_metric("./cognivoice/metrics.py", args.task)

        def new_compute_metrics(results):
            labels, label_mmse, sex_labels, lng_labels, pic_labels = results.label_ids
            if isinstance(results.predictions, tuple):
                logits, mmse_pred = results.predictions
            else:
                logits = results.predictions
                mmse_pred = None
            predictions = logits.argmax(-1)
            metrics = {}
            metrics['uar'] = recall_score(labels, predictions, average='macro')
            metrics['f1'] = f1_score(labels, predictions, average='binary')
            # metrics['f1'] = f1_score(labels, predictions)
            metrics['mse']  = mean_squared_error(label_mmse, mmse_pred)
            metrics['r2']  = r2_score(label_mmse, mmse_pred)
            for sex in set(sex_labels):
                sub_labels = labels[sex_labels == sex]
                sub_predictions = predictions[sex_labels == sex]
                sub_labels_mmse = label_mmse[sex_labels == sex]
                sub_predictions_mmse = mmse_pred[sex_labels == sex]
                metrics['mse_%s'%sex_map_rev[sex]] = mean_squared_error(sub_labels_mmse, sub_predictions_mmse)
                metrics['r2_%s'%sex_map_rev[sex]] = r2_score(sub_labels_mmse, sub_predictions_mmse)
            for lng in set(lng_labels):
                sub_labels = labels[lng_labels == lng]
                sub_predictions = predictions[lng_labels == lng]
                sub_labels_mmse = label_mmse[sex_labels == sex]
                sub_predictions_mmse = mmse_pred[sex_labels == sex]
                metrics['uar_%s'%lng_map_rev[lng]] = recall_score(sub_labels, sub_predictions, average='macro')
                metrics['f1_%s'%lng_map_rev[lng]] = f1_score(sub_labels, sub_predictions, average='binary')
                metrics['mse_%s'%lng_map_rev[lng]] = mean_squared_error(sub_labels_mmse, sub_predictions_mmse)
                metrics['r2_%s'%lng_map_rev[lng]] = r2_score(sub_labels_mmse, sub_predictions_mmse)
            return metrics

        def compute_metrics(p: EvalPrediction):
            # labels, label_mmse, sex_labels, lng_labels, pic_labels = p.label_ids
            # breakpoint()
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1) if args.task == 'cls' else np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
            result["combined_score"] = np.mean(list(result.values())).item()

            return result

        # Train
        # args.max_steps = 3
        # args.eval_steps = 1
        data_collator = default_data_collator if args.pad_to_max_length else None
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics,
            # tokenizer=tokenizer,
            # data_collator=data_collator,
        )

        # args.overwrite_output_dir = False
        # args.do_train = False
        # model.load_state_dict(ckpt)
        if args.do_train:
            # Detecting last checkpoint.
            last_checkpoint = None
            if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
                last_checkpoint = get_last_checkpoint(args.output_dir)
                if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
                    raise ValueError(
                        f"Output directory ({args.output_dir}) already exists and is not empty. "
                        "Use --overwrite_output_dir to overcome."
                    )
                elif last_checkpoint is not None and args.resume_from_checkpoint is None:
                    logger.info(
                        f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                    )
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            metrics = train_result.metrics

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if args.do_eval:
            logger.info("*** Evaluate ***")
            key = 'eval_' + args.metric_for_best_model
            eval_name = f'fold_{fold_id}'
            metrics = trainer.evaluate(eval_dataset=eval_data)
            trainer.log_metrics(eval_name, metrics)
            trainer.save_metrics(eval_name, metrics)

            scores.append(metrics[key])

            logger.info('*** Predict ***')
            predictions = trainer.predict(eval_data, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1) if args.task == 'cls' else np.squeeze(predictions)

            output_predict_file = os.path.join(output_dir_root, f"pred_{args.task}_fold_{fold_id}.csv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("idx,pred\n")
                    for i, p in zip(eval_idx, predictions):
                        writer.write(f'{i},{p}\n')


    wandb_log = {}
    for i, j in enumerate(scores):
        print(f'Fold: {i}, Score: {j:.4f}')
        wandb_log[f'per_fold/fold_{i}'] = j
    print('Mean score:', f'{np.mean(scores):.2f}')
    wandb_log['mean'] = np.mean(scores)
    wandb_log['std'] = np.std(scores)
    wandb_log['max'] = np.max(scores)
    wandb_log['min'] = np.min(scores)

    with open(os.path.join(output_dir_root, 'result.json'), 'w') as f:
        json.dump(wandb_log, f, indent=4)

    wandb.log(wandb_log)

if __name__ == "__main__":
    main()
