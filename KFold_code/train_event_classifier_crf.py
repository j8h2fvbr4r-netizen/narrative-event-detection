"""
Event classification: BERT/RoBERTa + CRF token classifier, 5-fold CV.

Converted from event_classification_final_fixed_crf.ipynb to run as a
batch job on any Slurm cluster. The model, the CRF trainer,
the tokenization/label-alignment logic, and the metrics are unchanged
from the notebook.

Training strategy: train + dev + test are ALL pooled into one dataset and
split into K folds (default 5), grouped by source_file so sentences from
the same document never end up split across train/held-out within a
fold, AND stratified by each sentence's dominant event label so class
distribution is balanced across folds as evenly as the grouping
constraint allows (StratifiedGroupKFold) -- plain GroupKFold ignores
labels entirely, which can leave some folds badly skewed if a class is
concentrated in just a few documents. A fresh model is trained on each
fold's K-1 remaining folds and evaluated on that fold's held-out slice
-- there is no separate, permanently-untouched test set anymore; every
sentence in the data acts as held-out evaluation data in exactly one
fold. Per-fold and mean +/- std metrics are reported and saved.

Usage (see also run_train.slurm for a matching sbatch script):

    python train_event_classifier_crf.py \
        --data_dir /scratch/$USER/event_annotation/classification \
        --language Italian \
        --n_splits 5 \
        --output_dir ./results \
        --model_save_dir ./best_model

Run `python train_event_classifier_crf.py --help` for all options.
"""

import argparse
import json
import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from evaluate import load as load_metric
from seqeval.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Data loading & grouping (token rows -> one row per sentence)
# --------------------------------------------------------------------------

def load_all_data(data_dir, languages):
    """
    Load train/dev/test CSVs for one or more languages and pool them into a
    single dataframe.

    Adds a 'language' column and a 'doc_id' column (language + '::' +
    source_file). doc_id exists specifically to disambiguate documents across
    languages when training multilingually -- source_file alone is NOT
    guaranteed unique across different languages' corpora (e.g. generic
    filenames like '5.txt' turning up in more than one language's data), and
    without this, sentences or fold-groups from two unrelated documents in
    different languages could silently get merged together. doc_id is used
    everywhere source_file was previously used as a grouping key.

    keep_default_na=False, na_values=[] is required here: pandas' default
    read_csv silently treats certain literal strings ('nan', 'NA', 'null',
    'N/A', etc.) as missing values on read, regardless of what they mean in
    the data. If a token column ever legitimately contains one of those
    words (e.g. 'nan' is a real word in some languages, including Bahasa
    Indonesia), the default behavior discards it and replaces it with an
    actual NaN -- silently corrupting real data, not just leaving a gap.
    Turning default NA-string detection off means a genuinely blank CSV
    cell comes through as an empty string instead, which is handled
    separately below.
    """
    frames = []
    for language in languages:
        for split in ["train", "dev", "test"]:
            df = pd.read_csv(
                os.path.join(data_dir, f"{split}_{language}.csv"),
                keep_default_na=False, na_values=[],
            )
            df["language"] = language
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["doc_id"] = full["language"] + "::" + full["source_file"].astype(str)

    if "token" in full.columns:
        blank = full["token"] == ""
        if blank.any():
            examples = full.loc[blank, ["language", "source_file", "sentence_id"]].head(5).to_dict("records")
            logger.warning(
                "%d rows have a genuinely empty 'token' cell in the source CSVs (not just a "
                "string pandas misread as missing -- keep_default_na=False rules that out). "
                "Replacing with a '[MISSING]' placeholder so word/label alignment stays intact "
                "rather than silently collapsing the sentence by one word. Worth checking the "
                "source CSVs directly. First few affected rows: %s",
                blank.sum(), examples,
            )
            full.loc[blank, "token"] = "[MISSING]"

    return full


def group_by_sentence(df, token_col, label2id):
    """
    Collapse one-row-per-token CSVs into one row per sentence.

    Groups by (doc_id, sentence_id) -- NOT (source_file, sentence_id) --
    since doc_id is the one guaranteed to be unique across languages (see
    load_all_data). Using source_file alone here in a multilingual pool
    would silently concatenate unrelated sentences from different languages
    that happen to share a filename and sentence_id into one fake sentence.

    sentence_text is rebuilt by joining the *actual annotated tokens* with
    single spaces (rather than reusing any pre-existing sentence_text
    column), so the whitespace-split word count always matches
    len(ner_tags) -- mismatches here are what caused the original
    IndexError this notebook/script was written to fix.
    """
    grouped = (
        df.groupby(["doc_id", "sentence_id"], sort=False)
        .apply(
            lambda g: pd.Series(
                {
                    "language": g["language"].iloc[0],
                    "source_file": g["source_file"].iloc[0],
                    "sentence_text": " ".join(g[token_col].fillna("[MISSING]").astype(str).tolist()),
                    "ner_tags": g["event_bio"].map(label2id).tolist(),
                }
            )
        )
        .reset_index()
    )
    return grouped



def report_mismatches(grouped, name):
    """
    Sanity check: whitespace-split word count should equal len(ner_tags)
    for every row. Non-empty output means the data has rows with e.g. a
    token containing an internal space -- fix the data, not the alignment
    code. tokenize_and_align_labels defensively drops overflow sub-tokens
    either way, but affected rows will have some unlabeled words.
    """
    mismatches = []
    for i, row in grouped.iterrows():
        n_words = len(row["sentence_text"].split())
        n_tags = len(row["ner_tags"])
        if n_words != n_tags:
            mismatches.append((i, row["source_file"], row["sentence_id"], n_words, n_tags))
    logger.info("%s: %d mismatched rows out of %d", name, len(mismatches), len(grouped))
    if mismatches:
        logger.info("First few mismatches: %s", mismatches[:10])
    return mismatches


def dominant_event_label(ner_tag_ids, id2label):
    """
    The most common non-O event type (suffix only, e.g. 'process', not
    'B-process'/'I-process') among a sentence's tokens -- used purely as a
    per-sentence stratification target for StratifiedGroupKFold, not as a
    training signal. Sentences with no event tokens at all stratify under
    a distinct 'O' bucket rather than being dropped or miscounted.
    """
    suffixes = [
        id2label[t].split("-", 1)[1]
        for t in ner_tag_ids
        if id2label[t] != "O"
    ]
    if not suffixes:
        return "O"
    return max(set(suffixes), key=suffixes.count)


# --------------------------------------------------------------------------
# 2. Tokenization & label alignment
# --------------------------------------------------------------------------

def make_tokenize_and_align_labels(tokenizer, max_length=512):
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["sentence_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        all_labels = []
        n_overflow = 0
        for i in range(len(examples["sentence_text"])):
            word_ids = tokenized.word_ids(batch_index=i)
            ner_tags = examples["ner_tags"][i]
            n_tags = len(ner_tags)

            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None or word_id == prev_word_id:
                    label_ids.append(-100)
                elif word_id < n_tags:
                    label_ids.append(ner_tags[word_id])
                else:
                    label_ids.append(-100)
                    n_overflow += 1
                prev_word_id = word_id

            all_labels.append(label_ids)

        if n_overflow:
            logger.warning(
                "[tokenize_and_align_labels] %d sub-tokens fell outside ner_tags range in this batch.",
                n_overflow,
            )

        tokenized["labels"] = all_labels
        return tokenized

    return tokenize_and_align_labels


# --------------------------------------------------------------------------
# 3. Model: encoder + linear classifier + CRF
# --------------------------------------------------------------------------

class ModelCRFForTokenClassification(nn.Module):
    """
    BERT/RoBERTa-based encoder + linear classifier + a CRF layer on top,
    to model label-sequence dependencies (e.g. discouraging an I-event
    from following an unrelated O) that plain per-token softmax ignores.

    The CRF only sees one timestep per *word* -- the first sub-token of
    each word, i.e. positions where labels != -100 -- not one timestep
    per sub-token. Subword continuation/padding positions would otherwise
    need fake "valid" tags just to satisfy the CRF, which would pollute
    the transition matrix it learns. _word_compaction does that
    full-sub-token -> compact-per-word conversion, driven entirely by
    `labels`, so no extra inputs are needed.
    """

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

    def _word_compaction(self, logits, labels):
        word_mask = labels != -100
        batch_size, _, _ = logits.shape
        max_words = max(int(word_mask.sum(dim=1).max().item()), 1)

        word_logits = logits.new_zeros((batch_size, max_words, self.num_labels))
        word_labels = labels.new_zeros((batch_size, max_words))
        word_pad_mask = torch.zeros((batch_size, max_words), dtype=torch.bool, device=logits.device)

        for b in range(batch_size):
            idx = word_mask[b].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n == 0:
                # degenerate empty sentence -- CRF requires mask[:, 0] all True
                word_pad_mask[b, 0] = True
                continue
            word_logits[b, :n] = logits[b, idx]
            word_labels[b, :n] = labels[b, idx]
            word_pad_mask[b, :n] = True
        return word_logits, word_labels, word_pad_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            word_logits, word_labels, word_pad_mask = self._word_compaction(logits, labels)
            loss = -self.crf(word_logits, word_labels, mask=word_pad_mask, reduction="mean")

        return TokenClassifierOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def decode(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Viterbi-decode the best label sequence per sentence, returned in
        the same (batch, seq_len) shape as `labels`, with -100 everywhere
        that isn't a first-sub-token-of-a-word position -- a drop-in
        replacement for np.argmax(logits, axis=2) in compute_metrics.
        """
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

    def save_pretrained(self, save_directory):
        """
        Mirrors HF's save_pretrained closely enough for the save-model step below.

        Writes the big torch.save binary (pytorch_model.bin) to a LOCAL temp
        directory first, then moves the finished file to save_directory with a
        plain file copy -- not directly via torch.save. This works around a
        repeated, deterministic corrupted-write bug when torch.save's zip
        writer targets /scratch directly (a networked filesystem): every
        failure so far has shown an "unexpected pos" gap of exactly 112 bytes,
        regardless of model/file size, which points to a filesystem/writer
        interaction bug rather than disk space or anything content-specific.
        A plain byte copy after the file already exists intact locally uses a
        completely different code path and sidesteps it.
        """
        os.makedirs(save_directory, exist_ok=True)

        local_tmp_dir = os.environ.get("TMPDIR") or tempfile.gettempdir()
        with tempfile.TemporaryDirectory(dir=local_tmp_dir) as staging_dir:
            local_weights_path = os.path.join(staging_dir, "pytorch_model.bin")
            torch.save(self.state_dict(), local_weights_path)
            shutil.copy2(local_weights_path, os.path.join(save_directory, "pytorch_model.bin"))

        self.config.save_pretrained(save_directory)
        with open(os.path.join(save_directory, "label_map.json"), "w") as f:
            json.dump({"id2label": self.id2label, "label2id": self.label2id}, f)

    @classmethod
    def from_pretrained(cls, load_directory, model_name="xlm-roberta-base"):
        with open(os.path.join(load_directory, "label_map.json")) as f:
            maps = json.load(f)
        id2label = {int(k): v for k, v in maps["id2label"].items()}
        label2id = maps["label2id"]
        model = cls(model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id)
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------

def make_compute_metrics(id2label):
    seqeval = load_metric("seqeval")

    def compute_metrics(p):
        preds, label_ids = p  # preds are already CRF-decoded label ids (see CRFTrainer)

        true_labels = [
            [id2label[l] for l in label_row if l != -100] for label_row in label_ids
        ]
        true_preds = [
            [id2label[pr] for pr, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(preds, label_ids)
        ]

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


# --------------------------------------------------------------------------
# 5. CRF-aware Trainer
# --------------------------------------------------------------------------

class CRFTrainer(Trainer):
    """
    Trainer subclass that swaps argmax-over-logits for CRF Viterbi
    decoding during evaluation/prediction, so compute_metrics receives
    actual best-path label sequences instead of raw per-token logits.
    """

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.detach() if has_labels else None
            decoded = model.decode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs.get("labels"),
            )

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        return (loss, decoded, labels)


# --------------------------------------------------------------------------
# 6. Per-fold train + evaluate
# --------------------------------------------------------------------------

def build_dataset(grouped_df, tokenizer, tokenize_and_align_labels):
    ds = Dataset.from_pandas(grouped_df.reset_index(drop=True))
    ds = ds.map(tokenize_and_align_labels, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    return ds


def decode_predictions(trainer, dataset, id2label):
    """Run trainer.predict and turn the CRF-decoded ids into string labels per sentence."""
    predictions = trainer.predict(dataset)
    preds = predictions.predictions  # already CRF-decoded label ids -- no argmax needed
    label_ids = predictions.label_ids

    true_labels, pred_labels = [], []
    for pred_row, label_row in zip(preds, label_ids):
        true_sent, pred_sent = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_sent.append(id2label[l])
            pred_sent.append(id2label[p])
        true_labels.append(true_sent)
        pred_labels.append(pred_sent)
    return true_labels, pred_labels


def word_level_records(grouped_df, true_labels, pred_labels, extra_cols=None):
    records = []
    extra_cols = extra_cols or {}
    n_truncated = 0
    for row_idx, (true_sent, pred_sent) in enumerate(zip(true_labels, pred_labels)):
        meta = grouped_df.iloc[row_idx]
        words = meta["sentence_text"].split()

        if len(true_sent) != len(pred_sent):
            # This isn't the truncation case below (that always keeps true/pred equal
            # length) -- something else is wrong, so fail loudly rather than guess.
            raise AssertionError(
                f"true/pred length mismatch at row {row_idx} (source_file={meta['source_file']!r}, "
                f"sentence_id={meta['sentence_id']!r}): {len(true_sent)} true vs {len(pred_sent)} pred. "
                f"This should never differ -- something upstream of word_level_records is broken."
            )

        if len(words) != len(true_sent):
            if len(true_sent) < len(words):
                # The tokenizer's max_length truncated this sentence -- everything past
                # the cutoff never got a sub-token, so it never got a label. Keep only
                # the words that actually have a prediction; log so it's visible rather
                # than silently dropping data.
                n_truncated += 1
                if n_truncated <= 5:
                    logger.warning(
                        "Row %d (source_file=%r, sentence_id=%r) was truncated by the "
                        "tokenizer's max_length: %d words in the source sentence, but only "
                        "%d got labeled. Keeping the first %d words; the rest have no "
                        "prediction. Consider raising --max_length if this happens often.",
                        row_idx, meta["source_file"], meta["sentence_id"],
                        len(words), len(true_sent), len(true_sent),
                    )
                words = words[: len(true_sent)]
            else:
                # More labels than words -- not explainable by truncation, and shouldn't
                # happen; fail loudly rather than silently misalign word<->label pairs.
                raise AssertionError(
                    f"length mismatch at row {row_idx} (source_file={meta['source_file']!r}, "
                    f"sentence_id={meta['sentence_id']!r}): {len(words)} words vs "
                    f"{len(true_sent)} true/pred labels -- more labels than words, which "
                    f"truncation cannot explain."
                )

        for word_idx, (word, true_label, pred_label) in enumerate(zip(words, true_sent, pred_sent)):
            rec = {
                "language": meta.get("language", None),
                "source_file": meta["source_file"],
                "sentence_id": meta["sentence_id"],
                "sentence_text": meta["sentence_text"],
                "word_index": word_idx,
                "word": word,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": true_label == pred_label,
            }
            rec.update(extra_cols)
            records.append(rec)
    if n_truncated:
        logger.warning(
            "%d / %d sentences in this fold's held-out set were truncated by the "
            "tokenizer's max_length -- their predictions cover only the first N words, "
            "not the full sentence.",
            n_truncated, len(true_labels),
        )
    return records


def run_fold(fold_idx, fold_train_grouped, fold_held_out_grouped, tokenizer,
             tokenize_and_align_labels, labels, id2label, label2id, args,
             base_model_short_name, language_tag):
    """
    Train on this fold's K-1 remaining folds, evaluate on its held-out
    fold. The held-out fold does double duty: it's what Trainer uses for
    epoch-level eval/best-model-selection AND the fold's reported result
    -- there is no further separate test set once the whole dataset is
    being folded.
    """
    logger.info("=" * 70)
    logger.info("FOLD %d/%d -- train sentences: %d | held-out sentences: %d",
                fold_idx + 1, args.n_splits, len(fold_train_grouped), len(fold_held_out_grouped))
    logger.info("=" * 70)

    train_dataset = build_dataset(fold_train_grouped, tokenizer, tokenize_and_align_labels)
    held_out_dataset = build_dataset(fold_held_out_grouped, tokenizer, tokenize_and_align_labels)

    model = ModelCRFForTokenClassification(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        dropout=args.dropout,
    )

    fp16 = torch.cuda.is_available() if args.fp16 is None else args.fp16
    fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # keep only the best checkpoint -- prevents /scratch quota blowout
                              # from accumulating a full checkpoint (model+optimizer state, easily
                              # 1-2GB+ each) per epoch per fold with nothing ever cleaned up
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=fp16,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = make_compute_metrics(id2label)

    trainer = CRFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=held_out_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    fold_results = trainer.evaluate(held_out_dataset)
    logger.info("Fold %d held-out metrics: %s", fold_idx, fold_results)

    true_labels, pred_labels = decode_predictions(trainer, held_out_dataset, id2label)
    logger.info("Fold %d held-out accuracy: %.4f", fold_idx, accuracy_score(true_labels, pred_labels))
    logger.info("Fold %d held-out report:\n%s",
                fold_idx, classification_report(true_labels, pred_labels, zero_division=0))

    # Save this fold's word-level held-out predictions
    records = word_level_records(fold_held_out_grouped, true_labels, pred_labels, extra_cols={"fold": fold_idx})
    predictions_df = pd.DataFrame(records)
    os.makedirs(args.predictions_dir, exist_ok=True)
    out_path = os.path.join(
        args.predictions_dir, f"res_{language_tag}_{base_model_short_name}_fold{fold_idx}.csv"
    )
    predictions_df.to_csv(out_path, index=False)
    logger.info("Fold %d: saved %d word-level held-out predictions to %s",
                fold_idx, len(predictions_df), out_path)

    # Save this fold's model
    fold_model_dir = os.path.join(
        args.model_save_dir or f"./best_model_{language_tag}_{base_model_short_name}_crf_cv",
        f"fold_{fold_idx}",
    )
    model.save_pretrained(fold_model_dir)
    tokenizer.save_pretrained(fold_model_dir)
    logger.info("Fold %d: model saved to %s", fold_idx, fold_model_dir)

    return {
        "fold": fold_idx,
        "n_train": len(fold_train_grouped),
        "n_held_out": len(fold_held_out_grouped),
        "precision": fold_results.get("eval_precision"),
        "recall": fold_results.get("eval_recall"),
        "f1": fold_results.get("eval_f1"),
        "accuracy": fold_results.get("eval_accuracy"),
    }


# --------------------------------------------------------------------------
# 7. Argument parsing
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing train_<language>.csv, dev_<language>.csv, test_<language>.csv "
                              "-- all three are loaded and pooled into one dataset for CV.")
    parser.add_argument("--language", type=str, default="Italian",
                         help="Single language to train on. Ignored if --languages is given.")
    parser.add_argument("--languages", type=str, nargs="+", default=None,
                         help="Two or more languages to pool and train ONE multilingual model on, e.g. "
                              "--languages Italian Dutch BI. Each gets train/dev/test loaded and pooled "
                              "together into a single dataset before CV. Documents are disambiguated "
                              "across languages internally (a 'doc_id' of language+source_file), so it's "
                              "safe even if two languages happen to reuse the same filename. Overrides "
                              "--language if given.")
    parser.add_argument("--token_col", type=str, default="token",
                         help="Column in the CSVs holding the individual token/word string per row.")

    # Cross-validation
    parser.add_argument("--n_splits", type=int, default=5,
                         help="Number of CV folds over the WHOLE dataset (train+dev+test pooled). Each "
                              "sentence is held out exactly once, across the n_splits folds.")
    parser.add_argument("--group_kfold", action="store_true", default=True,
                         help="Split folds grouped by --kfold_group_col (so a document's sentences never "
                              "straddle train/held-out within a fold) AND stratified by each sentence's "
                              "dominant event label, to balance class distribution across folds as much as "
                              "the grouping constraint allows (default: on).")
    parser.add_argument("--no_group_kfold", dest="group_kfold", action="store_false",
                         help="Use plain sentence-level KFold instead of grouping by document.")
    parser.add_argument("--kfold_group_col", type=str, default="doc_id",
                         help="Column used to group folds when --group_kfold is set. Default 'doc_id' "
                              "(language+source_file) is safe for both single- and multi-language runs; "
                              "only change this if you know what you're doing.")

    # Model
    parser.add_argument("--base_model", type=str, default="Musixmatch/umberto-commoncrawl-cased-v1",
                         help="HF model name/path used as the encoder.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512,
                         help="Max subword-token length per sentence (truncation cutoff). Sentences "
                              "longer than this get truncated -- their tail words won't have predictions "
                              "(handled gracefully, not a crash, but is real lost coverage). Raise this if "
                              "many sentences in your data are long; check the model's max supported "
                              "position/context length before raising it too far.")

    # Training
    parser.add_argument("--output_dir", type=str, default="./results",
                         help="Trainer output_dir (checkpoints, logs). Each fold gets its own subdirectory.")
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=None,
                         help="Force fp16 on. Default: auto (on if CUDA is available).")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false",
                         help="Force fp16 off.")

    # Outputs
    parser.add_argument("--predictions_dir", type=str, default="./predictions",
                         help="Where to save each fold's word-level held-out predictions CSV.")
    parser.add_argument("--model_save_dir", type=str, default=None,
                         help="Where to save fold models (under model_save_dir/fold_<i>). Default: "
                              "./best_model_<language>_<base_model_name>_crf_cv")
    parser.add_argument("--metrics_out", type=str, default=None,
                         help="Where to save the per-fold + mean/std summary CSV. Default: "
                              "<predictions_dir>/cv_summary_<language>_<base_model_name>.csv")

    return parser.parse_args()


# --------------------------------------------------------------------------
# 8. Main
# --------------------------------------------------------------------------

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
    # Used everywhere a single "language" string was previously used for filenames/logging --
    # e.g. "Italian" for a single-language run, "BI+Dutch+Italian" for a pooled multilingual one.
    language_tag = "+".join(sorted(languages))

    # -- 1. Load & group data (train + dev + test pooled into one dataset) ----
    logger.info("Loading data for language(s)=%s from %s (train+dev+test pooled)", languages, args.data_dir)
    full = load_all_data(args.data_dir, languages)

    labels = full["event_bio"].unique().tolist()
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info("Labels (%d): %s", len(labels), label2id)

    full_grouped = group_by_sentence(full, args.token_col, label2id)
    logger.info("Total pooled sentences (train+dev+test): %d", len(full_grouped))
    if len(languages) > 1:
        logger.info("Sentences per language: %s", full_grouped["language"].value_counts().to_dict())
    report_mismatches(full_grouped, "full (train+dev+test)")

    # -- 2. Tokenizer -----------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenize_and_align_labels = make_tokenize_and_align_labels(tokenizer, max_length=args.max_length)

    # -- 3. Build the K folds over the whole dataset -----------------------------
    # StratifiedGroupKFold balances each fold's dominant-event-label distribution
    # as well as it can *given* the document-grouping constraint -- it can't
    # perfectly balance a class concentrated in only a few documents (grouping
    # forces those documents to land whole in one fold each), but it's a
    # meaningful improvement over plain GroupKFold, which ignores labels
    # entirely when deciding which documents go in which fold.
    if args.group_kfold:
        n_groups = full_grouped[args.kfold_group_col].nunique()
        if n_groups < args.n_splits:
            raise ValueError(
                f"--group_kfold needs at least as many distinct '{args.kfold_group_col}' values as "
                f"--n_splits, but the pooled dataset only has {n_groups} "
                f"({sorted(full_grouped[args.kfold_group_col].unique())}) for --n_splits={args.n_splits}. "
                f"Either lower --n_splits, or pass --no_group_kfold to fall back to plain sentence-level "
                f"KFold (drops the same-document-never-splits guarantee)."
            )
        full_grouped["_dominant_label"] = full_grouped["ner_tags"].apply(
            lambda ids: dominant_event_label(ids, id2label)
        )
        logger.info("Dominant-label distribution (for fold stratification): %s",
                    full_grouped["_dominant_label"].value_counts().to_dict())

        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        groups = full_grouped[args.kfold_group_col]
        split_iter = splitter.split(full_grouped, y=full_grouped["_dominant_label"], groups=groups)
        logger.info("Using StratifiedGroupKFold(n_splits=%d) grouped by '%s', stratified by dominant "
                    "event label (%d distinct groups)", args.n_splits, args.kfold_group_col, n_groups)
    else:
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(full_grouped)
        logger.info("Using plain KFold(n_splits=%d), shuffled, seed=%d", args.n_splits, args.seed)

    fold_metrics = []
    for fold_idx, (train_idx, held_out_idx) in enumerate(split_iter):
        fold_train_grouped = full_grouped.iloc[train_idx]
        fold_held_out_grouped = full_grouped.iloc[held_out_idx]

        metrics = run_fold(
            fold_idx, fold_train_grouped, fold_held_out_grouped,
            tokenizer, tokenize_and_align_labels, labels, id2label, label2id, args,
            base_model_short_name, language_tag,
        )
        fold_metrics.append(metrics)

    # -- 4. Aggregate across folds -----------------------------------------------
    metrics_df = pd.DataFrame(fold_metrics)
    numeric_cols = [c for c in metrics_df.columns if c not in ("fold", "n_train", "n_held_out")]
    mean_row = metrics_df[numeric_cols].mean()
    std_row = metrics_df[numeric_cols].std()
    mean_row["fold"], std_row["fold"] = "mean", "std"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    logger.info("=" * 70)
    logger.info("%d-FOLD CV SUMMARY, whole dataset (language(s)=%s, model=%s)",
                args.n_splits, language_tag, base_model_short_name)
    logger.info("=" * 70)
    logger.info("\n%s", metrics_df.to_string(index=False))

    metrics_out = args.metrics_out or os.path.join(
        args.predictions_dir, f"cv_summary_{language_tag}_{base_model_short_name}.csv"
    )
    os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)
    logger.info("Saved CV summary to %s", metrics_out)


if __name__ == "__main__":
    main()
