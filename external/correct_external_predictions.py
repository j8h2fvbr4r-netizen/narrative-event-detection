"""
Apply the same BIO-correction heuristics (majority_suffix, all_B) used for
the k-fold out-of-fold predictions, but to the external Metamorphosis
test-set predictions from predict_and_evaluate_external.py instead.

Different situation from correct_oof_predictions.py's, so this is a separate
script rather than reusing that one directly:
  - That script combines several fold*.csv files into one non-overlapping
    partition (the whole point of k-fold OOF). Here there's no fold concept
    at all -- each language is already one complete, self-contained file
    (external_pred_<language>.csv) with gold_label/pred_label columns (not
    true_label/pred_label -- these came from a genuinely separate held-out
    document, not a CV fold).
  - This picks the best correction strategy on the POOLED external set
    across all given languages (matching how the multilingual model itself
    was trained pooled), and reports both the pooled result and a per-
    language breakdown, since you likely want to see both.

Imports the actual correction functions from correct_oof_predictions.py
rather than duplicating them, so there's one place to fix them if they
ever need adjusting.

Usage:

    python correct_external_predictions.py \
        --external_predictions_dir /scratch/$USER/event_crf/external_predictions \
        --languages Dutch Italian BI \
        --out_path /scratch/$USER/event_crf/external_predictions/external_pred_corrected.csv
"""

import argparse
import logging
import os

import pandas as pd
from seqeval.metrics import classification_report, f1_score

from correct_oof_predictions import CORRECTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_external_predictions(external_predictions_dir, languages, base_model_short_name):
    frames = []
    for language in languages:
        path = os.path.join(external_predictions_dir, f"external_pred_{language}_{base_model_short_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing external prediction file: {path}\n"
                f"Expected one file per language from predict_and_evaluate_external.py's --output_dir, "
                f"named external_pred_<language>_<base_model_short_name>.csv. Check --base_model_short_name "
                f"matches whatever predict_and_evaluate_external.py's --base_model produced."
            )
        df = pd.read_csv(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def reconstruct_sentences(df):
    """word-level rows -> one (gold_seq, pred_seq) pair per sentence, in
    word_index order, plus the sentence keys (language, source_file,
    sentence_id) in the same order."""
    gold_seqs, pred_seqs, keys = [], [], []
    for key, g in df.sort_values("word_index").groupby(["language", "source_file", "sentence_id"], sort=False):
        gold_seqs.append(g["gold_label"].tolist())
        pred_seqs.append(g["pred_label"].tolist())
        keys.append(key)
    return gold_seqs, pred_seqs, keys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--external_predictions_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, nargs="+", default=["Dutch", "Italian", "BI"])
    parser.add_argument("--base_model_short_name", type=str, required=True,
                         help="Must match the model name used in predict_and_evaluate_external.py's output "
                              "filenames, e.g. 'mmBERT-base' or 'xlm-roberta-base'.")
    parser.add_argument("--out_path", type=str, default=None,
                         help="Default: <external_predictions_dir>/external_pred_corrected.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_external_predictions(args.external_predictions_dir, args.languages, args.base_model_short_name)
    gold_seqs, pred_seqs, keys = reconstruct_sentences(df)
    logger.info("Loaded %d sentences across languages=%s", len(gold_seqs), args.languages)

    # -- Pick the best correction on the POOLED external set (all languages together) --
    candidates = {name: [fn(seq) for seq in pred_seqs] for name, fn in CORRECTIONS.items()}
    scores = {name: f1_score(gold_seqs, preds) for name, preds in candidates.items()}
    for name, score in scores.items():
        logger.info("  %-16s seqeval F1 (pooled, all languages) = %.4f", name, score)
    best_name = max(scores, key=scores.get)
    logger.info("Best correction on pooled external set: %s (F1=%.4f)", best_name, scores[best_name])

    logger.info("=== POOLED: before correction (raw) ===\n%s",
                classification_report(gold_seqs, candidates["raw"], zero_division=0))
    logger.info("=== POOLED: after correction (%s) ===\n%s",
                best_name, classification_report(gold_seqs, candidates[best_name], zero_division=0))

    # -- Also report per-language, using the SAME pooled-chosen correction (for a fair
    # like-for-like comparison across languages, not a separate strategy per language) --
    corrected_pred_seqs = candidates[best_name]
    for language in args.languages:
        lang_gold = [seq for (lang, *_), seq in zip(keys, gold_seqs) if lang == language]
        lang_pred_raw = [seq for (lang, *_), seq in zip(keys, candidates["raw"]) if lang == language]
        lang_pred_corrected = [seq for (lang, *_), seq in zip(keys, corrected_pred_seqs) if lang == language]
        if not lang_gold:
            continue
        # seqeval's classification_report crashes (ValueError: max() arg is empty) if a
        # subset has literally zero labeled entities of any kind -- guard rather than
        # let one thin/edge-case language kill the report for every other language too.
        try:
            logger.info("=== %s: before correction (raw) ===\n%s",
                        language, classification_report(lang_gold, lang_pred_raw, zero_division=0))
        except ValueError:
            logger.warning("%s: no entities found in gold/raw predictions -- skipping 'before' report.", language)
        try:
            logger.info("=== %s: after correction (%s) ===\n%s",
                        language, best_name, classification_report(lang_gold, lang_pred_corrected, zero_division=0))
        except ValueError:
            logger.warning("%s: no entities found in gold/corrected predictions -- skipping 'after' report.",
                            language)

    # -- Save the corrected predictions, word-level --
    records = []
    for (language, source_file, sentence_id), gold_seq, raw_seq, corrected_seq in zip(
        keys, gold_seqs, candidates["raw"], corrected_pred_seqs
    ):
        sentence_rows = df[
            (df["language"] == language) & (df["source_file"] == source_file) & (df["sentence_id"] == sentence_id)
        ].sort_values("word_index")
        words = sentence_rows["word"].tolist()
        for word_idx, (w, g, r, c) in enumerate(zip(words, gold_seq, raw_seq, corrected_seq)):
            records.append({
                "language": language,
                "source_file": source_file,
                "sentence_id": sentence_id,
                "word_index": word_idx,
                "word": w,
                "gold_label": g,
                "pred_label_raw": r,
                "pred_label_corrected": c,
                "correction_applied": best_name,
                "correct": g == c,
            })
    out_df = pd.DataFrame(records)

    out_path = args.out_path or os.path.join(
        args.external_predictions_dir, f"external_pred_corrected_{args.base_model_short_name}.csv"
    )
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %d word-level corrected external predictions to %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
