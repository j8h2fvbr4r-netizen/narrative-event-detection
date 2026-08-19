"""
Run the trained k-fold model ensemble on new, held-out documents and, since
this data already has gold event_bio labels (a curated annotation, not blind
text), evaluate against them as a genuine OUT-OF-DOMAIN external test set --
these documents were never part of train/dev/test/CV for any fold.

No retraining involved: this loads the already-saved fold models from
--model_dir (default: the multilingual mmBERT run) and ensembles their
predictions via per-token majority vote across folds.

Usage:

    python predict_and_evaluate_external.py \
        --model_dir /scratch/$USER/event_crf/best_model/BI+Dutch+Italian_mmBERT-base \
        --base_model "jhu-clsp/mmBERT-base" \
        --input_csv /scratch/$USER/event_crf/external/Metamorphosis_Dutch.csv \
        --language Dutch \
        --output_dir /scratch/$USER/event_crf/external_predictions
"""

import argparse
import glob
import json
import logging
import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from seqeval.metrics import classification_report as seq_report
from seqeval.scheme import IOB2
from sklearn.metrics import classification_report as sk_report
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Model class -- identical to train_event_classifier_crf.py's, needed here
# to load the saved fold weights back in. Only the parts inference actually
# uses are kept (decode + from_pretrained); training-only methods are omitted.
# --------------------------------------------------------------------------

class ModelCRFForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels, id2label, label2id, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.config = AutoConfig.from_pretrained(
            model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    @torch.no_grad()
    def decode(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        word_mask = labels != -100
        batch_size, seq_len, _ = logits.shape
        decoded = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=logits.device)

        for b in range(batch_size):
            idx = word_mask[b].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n == 0:
                continue
            word_logits = logits[b, idx].unsqueeze(0)
            mask = torch.ones((1, n), dtype=torch.bool, device=logits.device)
            best_path = self.crf.decode(word_logits, mask=mask)[0]
            for pos, lab in zip(idx.tolist(), best_path):
                decoded[b, pos] = lab
        return decoded

    @classmethod
    def from_pretrained(cls, load_directory, model_name):
        with open(os.path.join(load_directory, "label_map.json")) as f:
            maps = json.load(f)
        id2label = {int(k): v for k, v in maps["id2label"].items()}
        label2id = maps["label2id"]
        model = cls(model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id)
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, id2label, label2id


# --------------------------------------------------------------------------
# Data loading -- mirrors train_event_classifier_crf.py's conventions
# --------------------------------------------------------------------------

def normalize_label(label, label2id):
    """
    Fix known label-spelling inconsistencies seen in curated annotation
    exports (e.g. 'B-non event' with a literal space, instead of
    'B-non_event') before they can silently mismatch or crash evaluation.
    Logs once per distinct bad label found, not once per row.
    """
    if label in label2id:
        return label
    fixed = label.replace(" ", "_")
    if fixed in label2id:
        return fixed
    return label  # leave as-is; caller is responsible for surfacing unknown labels


def load_external_csv(path, label2id):
    df = pd.read_csv(path, keep_default_na=False, na_values=[])

    bad_labels = sorted(set(df["event_bio"]) - set(label2id.keys()))
    fixable = {l: normalize_label(l, label2id) for l in bad_labels}
    truly_unknown = {l: f for l, f in fixable.items() if f not in label2id}
    if fixable:
        logger.warning(
            "Found label(s) in the input not matching the model's known labels exactly: %s. "
            "Auto-corrected via space->underscore normalization where possible: %s. "
            "Still unrecognized after normalization (will break evaluation for these rows): %s",
            bad_labels, {k: v for k, v in fixable.items() if v != k}, list(truly_unknown.keys()),
        )
        df["event_bio"] = df["event_bio"].apply(lambda l: normalize_label(l, label2id))

    blank = df["token"] == ""
    if blank.any():
        logger.warning("%d rows have an empty token -- replacing with '[MISSING]'.", blank.sum())
        df.loc[blank, "token"] = "[MISSING]"

    grouped = (
        df.groupby(["source_file", "sentence_id"], sort=False)
        .apply(lambda g: pd.Series({
            "sentence_text": " ".join(g["token"].astype(str).tolist()),
            "words": g["token"].astype(str).tolist(),
            "gold_labels": g["event_bio"].tolist(),
        }))
        .reset_index()
    )
    return grouped


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def build_pseudo_labels(word_ids, device):
    """
    decode() needs a `labels`-shaped tensor purely to know which sub-token
    positions are the first sub-token of a word (labels != -100) -- it
    never looks at the actual label values, only whether a position is
    -100 or not. There are no real labels at inference time, so this
    builds a placeholder: 0 at each word's first sub-token, -100 everywhere
    else (continuation sub-tokens, special tokens, padding).
    """
    labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None or word_id == prev_word_id:
            labels.append(-100)
        else:
            labels.append(0)
        prev_word_id = word_id
    return torch.tensor([labels], device=device)


def predict_sentence_all_folds(models, tokenizer, sentence_text, n_words, max_length, device):
    """Run every fold model on one sentence, return each fold's predicted
    label id per word (truncation-aware: words beyond max_length get None)."""
    tokenized = tokenizer(sentence_text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    word_ids = tokenized.word_ids(batch_index=0)
    pseudo_labels = build_pseudo_labels(word_ids, device)

    fold_predictions = []  # one list per fold, each of length n_words (None if truncated away)
    for model in models:
        decoded = model.decode(
            input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], labels=pseudo_labels,
        )[0]
        # decoded has -100 everywhere except first-sub-token-of-word positions, in order
        word_preds = [d.item() for d in decoded if d.item() != -100]
        # pad with None if truncation dropped trailing words
        word_preds = word_preds + [None] * (n_words - len(word_preds))
        fold_predictions.append(word_preds)
    return fold_predictions


def ensemble_vote(fold_predictions, id2label):
    """Per word, majority vote across folds' predicted label ids. Ties broken
    by whichever label id is smallest (arbitrary but deterministic)."""
    n_words = len(fold_predictions[0])
    ensembled = []
    for w in range(n_words):
        votes = [fp[w] for fp in fold_predictions if fp[w] is not None]
        if not votes:
            ensembled.append(None)  # every fold truncated this word away
            continue
        counts = Counter(votes)
        best_count = max(counts.values())
        winners = sorted([v for v, c in counts.items() if c == best_count])
        ensembled.append(winners[0])
    return [id2label[l] if l is not None else "[TRUNCATED]" for l in ensembled]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", type=str, required=True,
                         help="Directory containing fold_0 .. fold_<n-1> subdirectories from training.")
    parser.add_argument("--base_model", type=str, required=True,
                         help="The HF encoder name used when this model was trained, e.g. 'jhu-clsp/mmBERT-base'. "
                              "Must match, or the saved weights won't load correctly.")
    parser.add_argument("--input_csv", type=str, required=True,
                         help="CSV in the same raw format as training data (source_file, sentence_id, token, "
                              "event_bio, ...). event_bio here is treated as GOLD for external evaluation.")
    parser.add_argument("--language", type=str, required=True,
                         help="Label for this input in output filenames/logging, e.g. 'Dutch'.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    base_model_short_name = args.base_model.split("/")[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Auto-detect: a k-fold CV run saves models under fold_0/, fold_1/, etc.
    # (this script ensembles across them). A --final_fit run saves ONE model
    # directly in model_dir, with no fold_* subdirectories at all -- in that
    # case, treat model_dir itself as the single model to load. Either way,
    # everything downstream (predict_sentence_all_folds/ensemble_vote) just
    # works on a models list, whether that list has 1 entry or several.
    fold_dirs = sorted(glob.glob(os.path.join(args.model_dir, "fold_*")))
    if fold_dirs:
        logger.info("Found %d fold models under %s -- ensembling across them.", len(fold_dirs), args.model_dir)
        model_dirs = fold_dirs
    elif os.path.exists(os.path.join(args.model_dir, "pytorch_model.bin")):
        logger.info("No fold_* subdirectories found -- treating %s as a single (e.g. final-fit) model.",
                     args.model_dir)
        model_dirs = [args.model_dir]
    else:
        raise FileNotFoundError(
            f"{args.model_dir} has neither fold_* subdirectories (k-fold CV models) nor a "
            f"pytorch_model.bin directly inside it (a single/final-fit model) -- check the path."
        )

    models, id2label_per_fold = [], []
    for model_dir in model_dirs:
        model, id2label, label2id = ModelCRFForTokenClassification.from_pretrained(model_dir, args.base_model)
        model.to(device)
        models.append(model)
        id2label_per_fold.append(id2label)
    # All folds should share the identical label set (built once from the whole
    # pooled dataset before folding) -- verify rather than assume.
    if len(set(tuple(sorted(m.items())) for m in id2label_per_fold)) != 1:
        raise ValueError(
            "Fold models have DIFFERENT label sets -- they don't look like they came from the same "
            "training run. Refusing to ensemble mismatched models."
        )
    id2label, label2id = id2label_per_fold[0], models[0].label2id
    logger.info("Labels (%d): %s", len(id2label), id2label)

    tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])

    logger.info("Loading external data: %s", args.input_csv)
    grouped = load_external_csv(args.input_csv, label2id)
    logger.info("Loaded %d sentences from %s", len(grouped), args.input_csv)

    records = []
    for _, row in grouped.iterrows():
        fold_preds = predict_sentence_all_folds(
            models, tokenizer, row["sentence_text"], len(row["words"]), args.max_length, device
        )
        ensembled_labels = ensemble_vote(fold_preds, id2label)
        for word_idx, (word, gold, pred) in enumerate(zip(row["words"], row["gold_labels"], ensembled_labels)):
            records.append({
                "language": args.language,
                "source_file": row["source_file"],
                "sentence_id": row["sentence_id"],
                "word_index": word_idx,
                "word": word,
                "gold_label": gold,
                "pred_label": pred,
                "correct": gold == pred,
            })

    out_df = pd.DataFrame(records)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"external_pred_{args.language}_{base_model_short_name}.csv")
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %d word-level predictions to %s", len(out_df), out_path)

    # -- Evaluate against gold, excluding any truncated tail (no prediction exists there) --
    scored = out_df[out_df["pred_label"] != "[TRUNCATED]"]
    n_truncated = len(out_df) - len(scored)
    if n_truncated:
        logger.warning("%d / %d words were truncated by max_length in every fold and have no prediction "
                        "-- excluded from the scores below.", n_truncated, len(out_df))

    def strip_bio(label):
        return "O" if label == "O" else label.split("-", 1)[1]

    flat_gold = [strip_bio(l) for l in scored["gold_label"]]
    flat_pred = [strip_bio(l) for l in scored["pred_label"]]
    logger.info("=== TOKEN-LEVEL (external test set: %s) ===\n%s",
                args.language, sk_report(flat_gold, flat_pred, zero_division=0))

    gold_seqs, pred_seqs = [], []
    for _, g in scored.sort_values("word_index").groupby(["source_file", "sentence_id"], sort=False):
        gold_seqs.append(g["gold_label"].tolist())
        pred_seqs.append(g["pred_label"].tolist())
    logger.info("=== SPAN-LEVEL, seqeval strict (external test set: %s) ===\n%s",
                args.language, seq_report(gold_seqs, pred_seqs, mode="strict", scheme=IOB2, zero_division=0))


if __name__ == "__main__":
    main()
