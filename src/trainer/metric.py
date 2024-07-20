from typing import Dict
import numpy as np
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


def decoder_metric_fn(pred: EvalPrediction) -> Dict:
    """
    LP: for common decoder plan
    """
    logits, labels = pred.predictions, pred.label_ids

    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    
    # 获得每个样本预测的排名，按概率降序排列
    ranked_indices = np.argsort(probabilities, axis=-1)[:, ::-1]
    
    # 找到正确答案的索引排名
    rank_of_correct = np.argwhere(ranked_indices == labels[:, None])[:, 1] + 1
    
    # 计算各个 hit@k 指标
    k_values = [1, 3, 5, 10]
    hit_at_k = {f'hit@{k}': np.mean(rank_of_correct <= k) for k in k_values}
    
    # 计算 MR (Mean Rank)
    mean_rank = np.mean(rank_of_correct)
    
    # 计算 MRR (Mean Reciprocal Rank)
    mean_reciprocal_rank = np.mean(1.0 / rank_of_correct)
    
    # 汇总所有指标
    metrics = {
        **hit_at_k,
        'MR': mean_rank,
        'MRR': mean_reciprocal_rank
    }

    return metrics
