"""Tests the training loop."""

from __future__ import annotations

import os
import shutil

import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from pylate import evaluation, losses, models, utils
from pylate.scores import XTRScores, colbert_scores


class _DummyModel:
    do_query_expansion = False


def _chunk_embeddings(embeddings: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    chunks = []
    start = 0
    for size in sizes:
        end = start + size
        chunks.append(embeddings[start:end])
        start = end
    return chunks


def _build_reps_and_masks(
    batch_size: int = 4,
    qt: int = 3,
    dt: int = 3,
    h: int = 4,
    chunk_sizes: list[int] | None = None,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    if chunk_sizes is None:
        chunk_sizes = [2, 2]

    anchor = torch.arange(batch_size * qt * h, dtype=torch.float32).view(
        batch_size, qt, h
    )
    positive = torch.arange(batch_size * dt * h, dtype=torch.float32).view(
        batch_size, dt, h
    )
    negative = torch.arange(batch_size * dt * h, dtype=torch.float32).view(
        batch_size, dt, h
    ) * 0.5

    reps = [
        _chunk_embeddings(anchor, chunk_sizes),
        _chunk_embeddings(positive, chunk_sizes),
        _chunk_embeddings(negative, chunk_sizes),
    ]

    masks = [
        torch.ones(batch_size, qt),
        torch.ones(batch_size, dt),
        torch.ones(batch_size, dt),
    ]
    return reps, masks


def test_contrastive_training() -> None:
    """Test constrastive training."""
    if os.path.exists(path="tests/cached_contrastive"):
        shutil.rmtree("tests/cached_contrastive")

    model = models.ColBERT(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

    dataset = load_dataset("lightonai/lighton-ms-marco-mini", "triplet", split="train")

    splits = dataset.train_test_split(test_size=0.5)

    train_dataset, eval_dataset = splits["train"], splits["test"]

    train_loss = losses.CachedContrastive(model=model, mini_batch_size=1)

    dev_evaluation = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    args = SentenceTransformerTrainingArguments(
        output_dir="tests/cached_contrastive",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        learning_rate=3e-6,
        do_eval=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluation,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train()

    model.save_pretrained("tests/cached_contrastive/final")

    assert os.path.isdir("tests/cached_contrastive")

    metrics = dev_evaluation(
        model=model,
        output_path="tests/cached_contrastive/",
    )

    assert isinstance(metrics, dict)

    assert os.path.isfile(
        path="tests/cached_contrastive/triplet_evaluation_results.csv"
    )

    results = pd.read_csv(
        filepath_or_buffer="tests/cached_contrastive/triplet_evaluation_results.csv"
    )

    assert "accuracy" in list(results.columns)

    if os.path.exists(path="tests/cached_contrastive"):
        shutil.rmtree("tests/cached_contrastive")


def test_score_mini_batch_size_consistency_non_full_batch() -> None:
    reps, masks = _build_reps_and_masks()

    loss_small = losses.CachedContrastive(
        model=_DummyModel(),
        score_metric=colbert_scores,
        mini_batch_size=2,
        score_mini_batch_size=1,
    ).calculate_loss(reps, masks)

    loss_large = losses.CachedContrastive(
        model=_DummyModel(),
        score_metric=colbert_scores,
        mini_batch_size=2,
        score_mini_batch_size=4,
    ).calculate_loss(reps, masks)

    assert torch.allclose(loss_small, loss_large, atol=1e-6, rtol=0)


def test_score_mini_batch_size_consistency_full_batch() -> None:
    reps, masks = _build_reps_and_masks()

    loss_small = losses.CachedContrastive(
        model=_DummyModel(),
        score_metric=XTRScores(k=4),
        mini_batch_size=2,
        score_mini_batch_size=1,
    ).calculate_loss(reps, masks)

    loss_large = losses.CachedContrastive(
        model=_DummyModel(),
        score_metric=XTRScores(k=4),
        mini_batch_size=2,
        score_mini_batch_size=4,
    ).calculate_loss(reps, masks)

    assert torch.allclose(loss_small, loss_large, atol=1e-6, rtol=0)
