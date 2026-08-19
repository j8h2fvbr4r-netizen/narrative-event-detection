"""
Train ONE deployment model on the ENTIRE labeled dataset -- every sentence,
nothing held out. Separate script from train_event_classifier_crf.py on
purpose: that script is for k-fold CV runs (the thing that tells you how well
a setup generalizes, and stays available for future monolingual/multilingual
CV experiments); this one is for producing the single model you'd actually
apply to new text, once CV has already told you the setup works.

Run k-fold CV first (train_event_classifier_crf.py, no changes needed) to get
a real generalization estimate -- report THAT number, not anything from this
script. This script's own end-of-training report is against the training set
itself and is explicitly NOT a generalization estimate; it only exists as a
sanity check that training actually worked.

Imports shared, already-hardened infrastructure (data loading with all its
bug fixes, the CRF model class with its save-then-copy fix for /scratch's
corrupted-write issue, the CRF-aware Trainer) directly from
train_event_classifier_crf.py rather than duplicating it -- keeps a single
source of truth so a future fix to data loading or model saving only needs
to happen in one place. train_final_fit.py must live in the same directory
as train_event_classifier_crf.py for this import to work.

Usage:

    python train_final_fit.py \
        --data_dir /scratch/$USER/event_crf/data \
        --languages Italian Dutch BI \
        --base_model "jhu-clsp/mmBERT-base" \
        --num_train_epochs 3 \
        --output_dir "$TMPDIR/results" \
        --predictions_dir /scratch/$USER/event_crf/predictions \
        --model_save_dir /scratch/$USER/event_crf/best_model
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import accuracy_score, classification_report
from transformers import DataCollatorForTokenClassification, TrainingArguments, AutoTokenizer

from train_event_classifier_crf import (
    load_all_data,
    group_by_sentence,
    report_mismatches,
    make_tokenize_and_align_labels,
    ModelCRFForTokenClassification,
    CRFTrainer,
    build_dataset,
    decode_predictions,
    word_level_records,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--language", type=str, default="Italian",
                         help="Single language to train on. Ignored if --languages is given.")
    parser.add_argument("--languages", type=str, nargs="+", default=None,
                         help="Two or more languages to pool into one multilingual model, e.g. "
                              "--languages Italian Dutch BI. Overrides --language if given.")
    parser.add_argument("--token_col", type=str, default="token")

    parser.add_argument("--base_model", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--output_dir", type=str, default="./results",
                         help="Trainer's own mid-training checkpoint dir. Point this at $TMPDIR on Habrok, "
                              "not /scratch -- see train_event_classifier_crf.py's notes on why.")
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")

    parser.add_argument("--predictions_dir", type=str, default="./predictions")
    parser.add_argument("--model_save_dir", type=str, default=None,
                         help="Final model is saved to <model_save_dir>/<language_tag>_<base_model_short>_final/")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    base_model_short_name = args.base_model.split("/")[-1]
    languages = args.languages if args.languages else [args.language]
    language_tag = "+".join(sorted(languages))

    # -- 1. Load & group ALL data -- no split, everything is training data ----
    logger.info("Loading data for language(s)=%s from %s (train+dev+test pooled, nothing held out)",
                languages, args.data_dir)
    full = load_all_data(args.data_dir, languages)

    labels = full["event_bio"].unique().tolist()
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info("Labels (%d): %s", len(labels), label2id)

    full_grouped = group_by_sentence(full, args.token_col, label2id)
    logger.info("Total sentences (100%% used for training, 0%% held out): %d", len(full_grouped))
    if len(languages) > 1:
        logger.info("Sentences per language: %s", full_grouped["language"].value_counts().to_dict())
    report_mismatches(full_grouped, "full (train+dev+test)")

    # -- 2. Tokenizer & dataset -------------------------------------------------
    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenize_and_align_labels = make_tokenize_and_align_labels(tokenizer, max_length=args.max_length)
    train_dataset = build_dataset(full_grouped, tokenizer, tokenize_and_align_labels)

    # -- 3. Train, no held-out eval ----------------------------------------------
    logger.info("=" * 70)
    logger.info("FINAL FIT -- training ONE model on all %d sentences, no held-out split", len(full_grouped))
    logger.info("=" * 70)

    model = ModelCRFForTokenClassification(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id, dropout=args.dropout,
    )

    fp16 = torch.cuda.is_available() if args.fp16 is None else args.fp16
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "final_fit"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="no",            # nothing held out to evaluate against
        save_strategy="epoch",         # crash-safety checkpoint to $TMPDIR only
        save_total_limit=1,
        load_best_model_at_end=False,  # no eval metric exists to pick a "best" epoch from
        fp16=fp16,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = CRFTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        processing_class=tokenizer, data_collator=data_collator,
    )

    trainer.train()
    logger.info("Final fit training complete (%.1f epochs on %d sentences).",
                args.num_train_epochs, len(full_grouped))

    # -- 4. Training-set sanity check ONLY -- NOT a generalization estimate ------
    true_labels, pred_labels = decode_predictions(trainer, train_dataset, id2label)
    logger.info(
        "Training-set fit check (NOT a generalization estimate -- report your earlier k-fold CV "
        "summary for this model's real expected performance, not this number): accuracy=%.4f\n%s",
        accuracy_score(true_labels, pred_labels),
        classification_report(true_labels, pred_labels, zero_division=0),
    )

    # -- 5. Save the final model --------------------------------------------------
    final_model_dir = os.path.join(
        args.model_save_dir or f"./best_model_{language_tag}_{base_model_short_name}",
        f"{language_tag}_{base_model_short_name}_final",
    )
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info("Final fit model saved to %s", final_model_dir)

    # -- 6. Save the training-set sanity-check predictions for reference ---------
    records = word_level_records(full_grouped, true_labels, pred_labels, extra_cols={"split": "train_sanity_check"})
    predictions_df = pd.DataFrame(records)
    os.makedirs(args.predictions_dir, exist_ok=True)
    out_path = os.path.join(
        args.predictions_dir, f"res_{language_tag}_{base_model_short_name}_final_fit_train_check.csv"
    )
    predictions_df.to_csv(out_path, index=False)
    logger.info("Saved %d word-level training-set sanity-check predictions to %s", len(predictions_df), out_path)


if __name__ == "__main__":
    main()
