from typing import Dict
import numpy as np
import evaluate
import torch
from transformers import EvalPrediction, AutoTokenizer
from sklearn.metrics import f1_score

from config.config import Config


class ROUGE:
    def __init__(self, cfg: Config, tokenizer: AutoTokenizer) -> None:
        # self.metric = evaluate.load("rouge", cache_dir=cfg.model.cache_dir)
        self.tokenizer = tokenizer

    def __call__(self, eval_preds: EvalPrediction):

        predictions, labels = eval_preds.predictions, eval_preds.label_ids

        delta_pad = abs(predictions.shape[1] - labels.shape[1])
        padding_value = 0  # 或者使用 -100
        padding = np.full(
            (predictions.shape[0], delta_pad),
            padding_value,
            dtype=predictions.dtype
        )

        # 在predictions前面插入填充向量
        if delta_pad > 0:
            predictions = np.concatenate([padding, predictions], axis=1)

        mask = labels != -100
        non_pad_predictions = predictions[mask]
        non_pad_labels = labels[mask]

        # 计算严格准确率
        all_tokens_exact_match_count = np.sum(non_pad_predictions == non_pad_labels)
        all_tokens_total_non_pad = np.sum(mask)
        all_tokens_exact_accuracy = (
            all_tokens_exact_match_count / all_tokens_total_non_pad
        )

        total_sample_cnt, correct_sample_cnt = 0, 0
        for _pred, _label in zip(predictions, labels):
            _mask = _label != -100
            __pred = _pred[_mask]
            __label = _label[_mask]

            if np.all(__pred == __label):
                correct_sample_cnt += 1
            total_sample_cnt += 1
        accuracy = correct_sample_cnt / total_sample_cnt

        return {
            "tokens_accuracy": all_tokens_exact_accuracy,
            "accuracy": accuracy,
            "total_sample_cnt": total_sample_cnt,
            "correct_sample_cnt": correct_sample_cnt,
        }


def metric_fn(pred: EvalPrediction) -> Dict:
    """
    TODO: This is for qa task.
    """
    squad_labels = pred.label_ids
    squad_preds = pred.predictions.argmax(-1)

    # Calculate Exact Match (EM)
    em = sum([1 if p == l else 0 for p, l in zip(squad_preds, squad_labels)]) / len(
        squad_labels
    )

    # Calculate F1-score
    f1 = f1_score(squad_labels, squad_preds, average="macro")

    return {"exact_match": em, "f1": f1}
