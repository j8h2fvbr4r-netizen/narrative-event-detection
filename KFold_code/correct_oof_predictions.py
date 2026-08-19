"""
Pick and apply a BIO-correction strategy using out-of-fold (OOF) k-fold
predictions, instead of picking it on one fixed dev split.

train_event_classifier_crf.py's k-fold run saves one prediction CSV per
fold (res_<language>_<model>_fold<i>.csv), each covering that fold's
held-out sentences -- predictions made by a model that never trained on
them. Concatenated across all folds, these give an out-of-fold prediction
for every single sentence in the dataset: an unbiased sample covering the
WHOLE corpus, not just whatever ended up in one arbitrary dev split.

This script:
  1. Loads and concatenates all fold prediction CSVs into one OOF set
     (asserting every sentence appears in exactly one fold -- if that's
     not true, something is wrong with how the folds were built/saved).
  2. Reconstructs per-sentence label sequences from the word-level rows.
  3. Scores three candidates on the full OOF set: raw predictions, and
     two suffix-correction heuristics (majority_suffix, all_B) that fix
     inconsistent event-type suffixes within a single continuous
     non-O span (something even a CRF can still get wrong, since it
     models B/I transitions but not a hard one-suffix-per-span constraint).
  4. Picks the best-scoring candidate and reports before/after metrics.
  5. Saves the OOF set with the winning correction applied -- this is
     the file to use for any downstream error analysis / evaluation
     notebook (confusion matrices, discontinuous-span analysis, etc.),
     since it covers 100% of the data with unbiased predictions.

Usage:

    python correct_oof_predictions.py \
        --predictions_dir ./predictions \
        --language Italian \
        --base_model_short_name umberto-commoncrawl-cased-v1 \
        --n_splits 5
"""

import argparse
import glob
import logging
import os

import pandas as pd
from seqeval.metrics import classification_report, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Correction heuristics (unchanged from predict_and_correct_aligned.ipynb)
# --------------------------------------------------------------------------

def fix_majority_suffix(labels):
    """Within each continuous non-O span, relabel every token with the
    suffix (event type) that occurs most often in that span."""
    labels = labels.copy()
    i = 0
    while i < len(labels):
        if labels[i] == "O":
            i += 1
            continue
        j = i
        while j < len(labels) and labels[j] != "O":
            j += 1
        suffixes = [l.split("-", 1)[1] for l in labels[i:j]]
        majority = max(set(suffixes), key=suffixes.count)
        labels[i] = f"B-{majority}"
        for k in range(i + 1, j):
            labels[k] = f"I-{majority}"
        i = j
    return labels


def fix_all_B(labels):
    """Within each continuous non-O span, adopt whichever suffix appears
    at the most recent B- tag (or the first token if the span starts
    with an I-)."""
    labels = labels.copy()
    current_suffix = None
    for i, l in enumerate(labels):
        if l == "O":
            current_suffix = None
            continue
        prefix, suffix = l.split("-", 1)
        if prefix == "B" or current_suffix is None:
            current_suffix = suffix
        labels[i] = f"{prefix}-{current_suffix}"
    return labels


CORRECTIONS = {
    "raw": lambda seq: seq,
    "majority_suffix": fix_majority_suffix,
    "all_B": fix_all_B,
}


# --------------------------------------------------------------------------
# Load & reconstruct sentences from the fold CSVs
# --------------------------------------------------------------------------

def load_oof_predictions(predictions_dir, language, base_model_short_name, n_splits):
    frames = []
    for fold_idx in range(n_splits):
        path = os.path.join(predictions_dir, f"res_{language}_{base_model_short_name}_fold{fold_idx}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing fold prediction file: {path}\n"
                f"Expected one file per fold (0..{n_splits - 1}) from train_event_classifier_crf.py's "
                f"--predictions_dir. Check --n_splits matches what training actually used."
            )
        df = pd.read_csv(path)
        frames.append(df)

    oof = pd.concat(frames, ignore_index=True)

    # Every (source_file, sentence_id) should appear in exactly one fold's file.
    sentence_fold_counts = oof.groupby(["source_file", "sentence_id"])["fold"].nunique()
    bad = sentence_fold_counts[sentence_fold_counts != 1]
    if len(bad):
        raise ValueError(
            f"{len(bad)} sentences appear in more than one fold's prediction file -- "
            f"the fold prediction CSVs don't look like a clean out-of-fold partition. "
            f"First few: {bad.index.tolist()[:5]}"
        )

    logger.info("Loaded OOF predictions: %d fold files, %d word-level rows, %d sentences",
                n_splits, len(oof), oof.groupby(["source_file", "sentence_id"]).ngroups)
    return oof


def reconstruct_sentences(oof_df):
    """word-level rows -> one (word_seq, true_seq, pred_seq) triple per
    sentence, in word_index order, plus the sentence keys in the same order."""
    word_seqs, true_seqs, pred_seqs, keys = [], [], [], []
    for key, g in oof_df.sort_values("word_index").groupby(["source_file", "sentence_id"], sort=False):
        word_seqs.append(g["word"].tolist())
        true_seqs.append(g["true_label"].tolist())
        pred_seqs.append(g["pred_label"].tolist())
        keys.append(key)
    return word_seqs, true_seqs, pred_seqs, keys


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions_dir", type=str, default="./predictions",
                         help="Directory containing the fold prediction CSVs from train_event_classifier_crf.py.")
    parser.add_argument("--language", type=str, default="Italian")
    parser.add_argument("--base_model_short_name", type=str, required=True,
                         help="The short model name used in the fold CSV filenames, "
                              "e.g. 'umberto-commoncrawl-cased-v1' (base_model.split('/')[-1]).")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--out_path", type=str, default=None,
                         help="Where to save the OOF predictions with the winning correction applied. "
                              "Default: <predictions_dir>/oof_<language>_<model>_corrected.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    oof = load_oof_predictions(args.predictions_dir, args.language, args.base_model_short_name, args.n_splits)
    word_seqs, true_seqs, pred_seqs, keys = reconstruct_sentences(oof)

    logger.info("Scoring correction candidates on the full out-of-fold set (%d sentences)...", len(true_seqs))
    candidates = {name: [fn(seq) for seq in pred_seqs] for name, fn in CORRECTIONS.items()}
    scores = {name: f1_score(true_seqs, preds) for name, preds in candidates.items()}
    for name, score in scores.items():
        logger.info("  %-16s seqeval F1 = %.4f", name, score)

    best_name = max(scores, key=scores.get)
    logger.info("Best correction on OOF data: %s (F1=%.4f)", best_name, scores[best_name])

    logger.info("=== Before correction (raw OOF predictions) ===\n%s",
                classification_report(true_seqs, candidates["raw"], zero_division=0))
    logger.info("=== After correction (%s) ===\n%s",
                best_name, classification_report(true_seqs, candidates[best_name], zero_division=0))

    # -- Save the corrected OOF predictions, word-level, for downstream use --
    corrected_pred_seqs = candidates[best_name]
    records = []
    for (source_file, sentence_id), word_seq, true_seq, raw_seq, corrected_seq in zip(
        keys, word_seqs, true_seqs, candidates["raw"], corrected_pred_seqs
    ):
        for word_idx, (w, t, r, c) in enumerate(zip(word_seq, true_seq, raw_seq, corrected_seq)):
            records.append({
                "source_file": source_file,
                "sentence_id": sentence_id,
                "word_index": word_idx,
                "word": w,
                "true_label": t,
                "pred_label_raw": r,
                "pred_label_corrected": c,
                "correction_applied": best_name,
                "correct": t == c,
            })
    out_df = pd.DataFrame(records)

    out_path = args.out_path or os.path.join(
        args.predictions_dir, f"oof_{args.language}_{args.base_model_short_name}_corrected.csv"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %d word-level OOF rows (with '%s' correction applied) to %s",
                len(out_df), best_name, out_path)


if __name__ == "__main__":
    main()
