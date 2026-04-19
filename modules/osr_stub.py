from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from schemas import AnomalyResult, EventRecord, OSRResult, RuleMatchResult


class CharTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)              # [B, L, E]
        emb = emb.transpose(1, 2)            # [B, E, L]

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))
            p = torch.max(c, dim=2).values
            conv_outs.append(p)

        feat = torch.cat(conv_outs, dim=1)
        feat = self.dropout(feat)
        feat = torch.relu(self.fc1(feat))
        logits = self.classifier(feat)
        return logits, feat


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(ch, vocab.get("<unk>", 1)) for ch in str(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab.get("<pad>", 0)] * (max_len - len(ids))
    return ids


class OSREngine:
    def __init__(self) -> None:
        self.method = "textcnn_osr_hardneg75"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ckpt = Path(
            os.getenv("SHIELD_OSR_MODEL_CKPT", "checkpoints/textcnn_closedset.pt")
        )
        self.osr_ckpt = Path(
            os.getenv("SHIELD_OSR_ARTIFACT_CKPT", "checkpoints/osr_artifacts.pt")
        )

        self.model = None
        self.vocab = None
        self.id2label = None
        self.max_len = 256
        self.centroids = None
        self.thresholds = None
        self._ready = False
        self._init_error = ""

        self._load()

    def _load(self) -> None:
        try:
            if not self.model_ckpt.exists():
                raise FileNotFoundError(f"missing model ckpt: {self.model_ckpt}")
            if not self.osr_ckpt.exists():
                raise FileNotFoundError(f"missing osr ckpt: {self.osr_ckpt}")

            model_bundle = torch.load(self.model_ckpt, map_location=self.device)
            osr_bundle = torch.load(self.osr_ckpt, map_location=self.device)

            self.vocab = model_bundle["vocab"]
            raw_id2label = model_bundle["id2label"]
            self.id2label = {int(k): v for k, v in raw_id2label.items()} if isinstance(next(iter(raw_id2label.keys())), str) else raw_id2label

            config = model_bundle["config"]
            self.max_len = int(config["max_len"])

            self.model = CharTextCNN(
                vocab_size=len(self.vocab),
                num_classes=len(model_bundle["label2id"]),
                emb_dim=int(config["emb_dim"]),
                num_filters=int(config["num_filters"]),
                hidden_dim=int(config["hidden_dim"]),
                dropout=float(config["dropout"]),
                pad_idx=self.vocab["<pad>"],
            ).to(self.device)
            self.model.load_state_dict(model_bundle["model_state"])
            self.model.eval()

            if "thresholds" in osr_bundle:
                self.thresholds = osr_bundle["thresholds"]
            else:
                self.thresholds = osr_bundle

            if "centroids" in osr_bundle:
                raw_centroids = osr_bundle["centroids"]
            elif "class_centroids" in osr_bundle:
                raw_centroids = osr_bundle["class_centroids"]
            else:
                raise KeyError("osr artifact missing centroids/class_centroids")

            self.centroids = self._normalize_centroids(raw_centroids)
            self.thresholds = self._normalize_thresholds(self.thresholds)

            self._ready = True
        except Exception as e:
            self._ready = False
            self._init_error = str(e)

    def _normalize_centroids(self, raw):
        if isinstance(raw, dict):
            out = {}
            for k, v in raw.items():
                out[int(k)] = np.asarray(v, dtype=np.float32)
            return out
        if isinstance(raw, (list, tuple)):
            return {i: np.asarray(v, dtype=np.float32) for i, v in enumerate(raw)}
        raise TypeError("unsupported centroid structure")

    def _normalize_thresholds(self, thresholds: dict) -> dict:
        out = dict(thresholds)
        class_th = out.get("class_distance_thresholds", {})
        if isinstance(class_th, dict):
            out["class_distance_thresholds"] = {int(k): float(v) for k, v in class_th.items()}
        return out

    def _fallback_recognize(
        self,
        anomaly_result: AnomalyResult,
        rule_result: RuleMatchResult,
    ) -> OSRResult:
        if rule_result.hit:
            payload = {
                "pred_label": rule_result.rule_name or "known_suspicious",
                "final_label": rule_result.rule_name or "known_suspicious",
                "max_prob": 0.95,
                "dist_to_pred_centroid": None,
                "prob_threshold": None,
                "class_distance_threshold": None,
                "is_unknown": 0,
                "reject_reason": "rule_already_matched",
                "engine_ready": False,
                "engine_error": self._init_error,
            }
            return OSRResult(
                is_unknown=False,
                confidence=0.95,
                method="adaptive_osr_stub_fallback",
                reason=json.dumps(payload, ensure_ascii=False),
            )

        threshold = max(float(anomaly_result.threshold) + 0.05, 0.50)
        if anomaly_result.model_name.startswith("webhawk"):
            threshold = max(threshold, 0.70)

        strong_cluster_signal = any(
            token in anomaly_result.reason for token in ["noise_point=-1", "minority_cluster=1"]
        )
        is_unknown = bool(
            anomaly_result.is_anomaly and
            (anomaly_result.score >= threshold or strong_cluster_signal)
        )

        payload = {
            "pred_label": "benign",
            "final_label": "unknown" if is_unknown else "benign",
            "max_prob": float(anomaly_result.score),
            "dist_to_pred_centroid": None,
            "prob_threshold": threshold,
            "class_distance_threshold": None,
            "is_unknown": int(is_unknown),
            "reject_reason": (
                f"anomaly score {anomaly_result.score:.2f} >= unknown threshold {threshold:.2f}"
                if is_unknown else
                f"anomaly score {anomaly_result.score:.2f} < unknown threshold {threshold:.2f}"
            ),
            "engine_ready": False,
            "engine_error": self._init_error,
        }
        return OSRResult(
            is_unknown=is_unknown,
            confidence=min(0.95, max(0.60, float(anomaly_result.score))),
            method="adaptive_osr_stub_fallback",
            reason=json.dumps(payload, ensure_ascii=False),
        )

    def _canonicalize_event_text(self, event: EventRecord) -> str:
        raw = str(getattr(event, "raw_text", "") or "").strip()
        if "METHOD=" in raw and "PATH=" in raw and "PROTOCOL=" in raw:
            return raw

        method = getattr(event, "method", None) or getattr(event, "http_method", None)
        path = getattr(event, "path", None) or getattr(event, "request_path", None)
        query = getattr(event, "query", None) or getattr(event, "query_string", None) or ""
        protocol = getattr(event, "protocol", None) or getattr(event, "http_version", None)

        if not method or not path:
            # 兼容 access log 中的 "GET /path?a=1 HTTP/1.1"
            m = re.search(r'"([A-Z]+)\s+(\S+)\s+(HTTP/\d\.\d)"', raw)
            if not m:
                m = re.search(r'^([A-Z]+)\s+(\S+)\s+(HTTP/\d\.\d)$', raw)

            if m:
                method = method or m.group(1)
                target = m.group(2)
                protocol = protocol or m.group(3)
                if "?" in target:
                    path = path or target.split("?", 1)[0]
                    query = query or target.split("?", 1)[1]
                else:
                    path = path or target

        method = str(method or "GET")
        path = str(path or "/")
        query = str(query or "")
        protocol = str(protocol or "HTTP/1.1")

        return f"METHOD={method} PATH={path} QUERY={query} PROTOCOL={protocol}"

    def _infer_one(self, text: str) -> dict:
        x = torch.tensor([encode_text(text, self.vocab, self.max_len)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, feat = self.model(x)
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_id = torch.max(probs, dim=1)

        pred_id = int(pred_id.item())
        pred_label = self.id2label[pred_id]
        max_prob = float(max_prob.item())
        emb = feat[0].detach().cpu().numpy().astype(np.float32)

        dist_to_pred_centroid = float(np.linalg.norm(emb - self.centroids[pred_id]))
        base_prob_th = float(self.thresholds["global_prob_threshold"])
        base_dist_th = float(self.thresholds["class_distance_thresholds"][pred_id])

        prob_th = base_prob_th
        dist_th = base_dist_th

        # 采用你最后选出的 hardneg75 对应策略
        if pred_label == "command_exec":
            prob_th = max(base_prob_th, 0.75)
            dist_th = min(base_dist_th, base_dist_th * 0.85)

        low_conf = max_prob < prob_th
        far_from_centroid = dist_to_pred_centroid > dist_th

        if pred_label == "benign":
            # benign 更宽松：必须同时低置信且远离中心才拒识
            if low_conf and far_from_centroid:
                final_label = "unknown"
                reject_reason = "low_confidence+far_from_centroid"
                is_unknown = 1
            else:
                final_label = "benign"
                reject_reason = "accepted"
                is_unknown = 0
        else:
            # 攻击类保持严格
            if low_conf and far_from_centroid:
                final_label = "unknown"
                reject_reason = "low_confidence+far_from_centroid"
                is_unknown = 1
            elif low_conf:
                final_label = "unknown"
                reject_reason = "low_confidence"
                is_unknown = 1
            elif far_from_centroid:
                final_label = "unknown"
                reject_reason = "far_from_centroid"
                is_unknown = 1
            else:
                final_label = pred_label
                reject_reason = "accepted"
                is_unknown = 0

        return {
            "pred_label": pred_label,
            "final_label": final_label,
            "max_prob": max_prob,
            "dist_to_pred_centroid": dist_to_pred_centroid,
            "prob_threshold": prob_th,
            "class_distance_threshold": dist_th,
            "is_unknown": is_unknown,
            "reject_reason": reject_reason,
            "engine_ready": True,
            "engine_error": "",
        }

    def recognize(
        self,
        event: EventRecord,
        anomaly_result: AnomalyResult,
        rule_result: RuleMatchResult,
    ) -> OSRResult:
        if rule_result.hit:
            payload = {
                "pred_label": rule_result.rule_name or "known_suspicious",
                "final_label": rule_result.rule_name or "known_suspicious",
                "max_prob": 0.95,
                "dist_to_pred_centroid": None,
                "prob_threshold": None,
                "class_distance_threshold": None,
                "is_unknown": 0,
                "reject_reason": "rule_already_matched",
                "engine_ready": self._ready,
                "engine_error": self._init_error,
            }
            return OSRResult(
                is_unknown=False,
                confidence=0.95,
                method=self.method,
                reason=json.dumps(payload, ensure_ascii=False),
            )

        if not self._ready:
            return self._fallback_recognize(anomaly_result, rule_result)

        try:
            text = self._canonicalize_event_text(event)
            result = self._infer_one(text)
            return OSRResult(
                is_unknown=bool(result["is_unknown"]),
                confidence=float(result["max_prob"]),
                method=self.method,
                reason=json.dumps(result, ensure_ascii=False),
            )
        except Exception as e:
            self._init_error = f"inference_error: {e}"
            return self._fallback_recognize(anomaly_result, rule_result)