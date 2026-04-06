import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SplitBundle:
    train: Tuple[torch.Tensor, torch.Tensor]
    val: Tuple[torch.Tensor, torch.Tensor]
    test_iid: Tuple[torch.Tensor, torch.Tensor]
    test_shift: Tuple[torch.Tensor, torch.Tensor]
    correction: Tuple[torch.Tensor, torch.Tensor]
    correction_eval: Tuple[torch.Tensor, torch.Tensor]


class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        last_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class TeachabilityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        last_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_width
        self.backbone = nn.Sequential(*layers)
        self.task_head = nn.Linear(last_dim, 1)
        self.correction_head = nn.Linear(last_dim, 1, bias=False)
        nn.init.zeros_(self.correction_head.weight)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        feats = self.features(x)
        logits = self.task_head(feats)
        if use_correction:
            logits = logits + self.correction_head(feats)
        return logits.squeeze(-1)


def make_loader(split: Tuple[torch.Tensor, torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(split[0], split[1]), batch_size=batch_size, shuffle=shuffle)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def evaluate_classifier(
    model: nn.Module,
    split: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    loader = make_loader(split, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).float().sum().item()
            total_count += x.size(0)
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def fit_classifier(
    model: nn.Module,
    train_split: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str = "adamw",
) -> nn.Module:
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    loader = make_loader(train_split, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model


def evaluate_all_splits(
    model: nn.Module,
    bundle: SplitBundle,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    train_metrics = evaluate_classifier(model, bundle.train, device, batch_size=batch_size)
    val_metrics = evaluate_classifier(model, bundle.val, device, batch_size=batch_size)
    test_metrics = evaluate_classifier(model, bundle.test_iid, device, batch_size=batch_size)
    shift_metrics = evaluate_classifier(model, bundle.test_shift, device, batch_size=batch_size)
    metrics = {
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
        "test_iid_loss": test_metrics["loss"],
        "test_iid_acc": test_metrics["acc"],
        "test_shift_loss": shift_metrics["loss"],
        "test_shift_acc": shift_metrics["acc"],
        "gen_gap": test_metrics["loss"] - train_metrics["loss"],
    }
    return metrics


def residual_logits(model: nn.Module, adapter: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feats = model.features(x)
        if isinstance(model, TeachabilityMLP):
            base_logits = model.task_head(feats)
        else:
            base_logits = model.head(feats)
    return (base_logits + adapter(feats)).squeeze(-1)


def evaluate_residual_classifier(
    model: nn.Module,
    adapter: nn.Module,
    split: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    loader = make_loader(split, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    adapter.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = residual_logits(model, adapter, x)
            loss = criterion(logits, y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).float().sum().item()
            total_count += x.size(0)
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def make_frozen_projector(in_features: int, out_features: int, device: torch.device) -> nn.Linear:
    projector = nn.Linear(in_features, out_features, bias=False).to(device)
    nn.init.normal_(projector.weight, mean=0.0, std=1.0 / max(in_features, 1) ** 0.5)
    for parameter in projector.parameters():
        parameter.requires_grad = False
    return projector


def projected_residual_logits(model: nn.Module, projector: nn.Module, adapter: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feats = model.features(x)
        projected = projector(feats)
        if isinstance(model, TeachabilityMLP):
            base_logits = model.task_head(feats)
        else:
            base_logits = model.head(feats)
    return (base_logits + adapter(projected)).squeeze(-1)


def evaluate_projected_residual_classifier(
    model: nn.Module,
    projector: nn.Module,
    adapter: nn.Module,
    split: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    loader = make_loader(split, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    projector.eval()
    adapter.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = projected_residual_logits(model, projector, adapter, x)
            loss = criterion(logits, y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).float().sum().item()
            total_count += x.size(0)
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def correction_audit(
    model: nn.Module,
    correction_split: Tuple[torch.Tensor, torch.Tensor],
    correction_eval_split: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    adapter_rank: int = 8,
    weight_decay: float = 0.0,
) -> Dict[str, float]:
    audited = clone_model(model).to(device)
    for parameter in audited.parameters():
        parameter.requires_grad = False
    projector = make_frozen_projector(audited.head.in_features, adapter_rank, device)
    adapter = nn.Linear(adapter_rank, 1, bias=False).to(device)
    nn.init.zeros_(adapter.weight)
    before = evaluate_projected_residual_classifier(audited, projector, adapter, correction_eval_split, device, batch_size=batch_size)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(correction_split, batch_size=batch_size, shuffle=True)
    batches = list(loader)
    audited.eval()
    train_before = evaluate_projected_residual_classifier(audited, projector, adapter, correction_split, device, batch_size=batch_size)
    before_params = [parameter.detach().clone() for parameter in adapter.parameters()]
    adapter.train()
    for step_idx in range(steps):
        x, y = batches[step_idx % len(batches)]
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = projected_residual_logits(audited, projector, adapter, x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    train_after = evaluate_projected_residual_classifier(audited, projector, adapter, correction_split, device, batch_size=batch_size)
    after = evaluate_projected_residual_classifier(audited, projector, adapter, correction_eval_split, device, batch_size=batch_size)
    param_shift_sq = 0.0
    for before_param, after_param in zip(before_params, adapter.parameters()):
        delta = after_param.detach() - before_param
        param_shift_sq += float(torch.sum(delta * delta).item())
    return {
        "correction_train_pre_acc": train_before["acc"],
        "correction_train_post_acc": train_after["acc"],
        "correction_train_gain": train_after["acc"] - train_before["acc"],
        "correction_train_pre_loss": train_before["loss"],
        "correction_train_post_loss": train_after["loss"],
        "correction_eval_pre_acc": before["acc"],
        "correction_eval_post_acc": after["acc"],
        "correction_eval_gain": after["acc"] - before["acc"],
        "correction_eval_pre_loss": before["loss"],
        "correction_eval_post_loss": after["loss"],
        "correction_param_shift_l2": param_shift_sq ** 0.5,
        "correction_pre_acc": before["acc"],
        "correction_post_acc": after["acc"],
        "correction_gain": after["acc"] - before["acc"],
        "correction_pre_loss": before["loss"],
        "correction_post_loss": after["loss"],
    }


def teachability_probe(
    model: TeachabilityMLP,
    correction_split: Tuple[torch.Tensor, torch.Tensor],
    correction_eval_split: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    steps: int,
    lr: float,
    lr_scale: float,
    batch_size: int,
) -> Dict[str, float]:
    audited = clone_model(model).to(device)
    for parameter in audited.backbone.parameters():
        parameter.requires_grad = False
    for parameter in audited.task_head.parameters():
        parameter.requires_grad = False
    for parameter in audited.correction_head.parameters():
        parameter.requires_grad = True
    before = evaluate_residual_classifier(audited, audited.correction_head, correction_eval_split, device, batch_size=batch_size)
    optimizer = torch.optim.SGD(audited.correction_head.parameters(), lr=lr * lr_scale, momentum=0.0)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(correction_split, batch_size=batch_size, shuffle=True)
    batches = list(loader)
    audited.eval()
    train_before = evaluate_residual_classifier(audited, audited.correction_head, correction_split, device, batch_size=batch_size)
    before_params = [parameter.detach().clone() for parameter in audited.correction_head.parameters()]
    audited.correction_head.train()
    for step_idx in range(steps):
        x, y = batches[step_idx % len(batches)]
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = residual_logits(audited, audited.correction_head, x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    train_after = evaluate_residual_classifier(audited, audited.correction_head, correction_split, device, batch_size=batch_size)
    after = evaluate_residual_classifier(audited, audited.correction_head, correction_eval_split, device, batch_size=batch_size)
    param_shift_sq = 0.0
    for before_param, after_param in zip(before_params, audited.correction_head.parameters()):
        delta = after_param.detach() - before_param
        param_shift_sq += float(torch.sum(delta * delta).item())
    return {
        "probe_train_pre_acc": train_before["acc"],
        "probe_train_post_acc": train_after["acc"],
        "probe_train_gain": train_after["acc"] - train_before["acc"],
        "probe_train_pre_loss": train_before["loss"],
        "probe_train_post_loss": train_after["loss"],
        "probe_eval_pre_acc": before["acc"],
        "probe_eval_post_acc": after["acc"],
        "probe_eval_gain": after["acc"] - before["acc"],
        "probe_eval_pre_loss": before["loss"],
        "probe_eval_post_loss": after["loss"],
        "probe_param_shift_l2": param_shift_sq ** 0.5,
        "probe_pre_acc": before["acc"],
        "probe_post_acc": after["acc"],
        "probe_gain": after["acc"] - before["acc"],
        "probe_pre_loss": before["loss"],
        "probe_post_loss": after["loss"],
        "probe_train_acc": train_after["acc"],
    }


def fit_task_only(
    model: TeachabilityMLP,
    train_split: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> TeachabilityMLP:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True
    for parameter in model.task_head.parameters():
        parameter.requires_grad = True
    for parameter in model.correction_head.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    loader = make_loader(train_split, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model


def evaluate_teachability_model(
    model: TeachabilityMLP,
    bundle: SplitBundle,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    train_metrics = evaluate_classifier(model, bundle.train, device, batch_size=batch_size)
    val_metrics = evaluate_classifier(model, bundle.val, device, batch_size=batch_size)
    test_metrics = evaluate_classifier(model, bundle.test_iid, device, batch_size=batch_size)
    shift_metrics = evaluate_classifier(model, bundle.test_shift, device, batch_size=batch_size)
    return {
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
        "test_iid_loss": test_metrics["loss"],
        "test_iid_acc": test_metrics["acc"],
        "test_shift_loss": shift_metrics["loss"],
        "test_shift_acc": shift_metrics["acc"],
    }


def make_shortcut_split(
    n: int,
    input_dim: int,
    shortcut_strength: float,
    shortcut_alignment: float,
    label_noise: float,
    feature_noise: float,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    latent = rng.normal(size=(n, 3)).astype(np.float32)
    base_score = 1.25 * latent[:, 0] - 0.9 * latent[:, 1] + 0.5 * latent[:, 2]
    clean_y = (base_score > 0.0).astype(np.float32)
    flip_mask = rng.random(n) < label_noise
    y = clean_y.copy()
    y[flip_mask] = 1.0 - y[flip_mask]

    x = np.zeros((n, input_dim), dtype=np.float32)
    x[:, 0] = base_score + feature_noise * rng.normal(size=n)
    x[:, 1] = latent[:, 0] - latent[:, 1] + feature_noise * rng.normal(size=n)
    x[:, 2] = shortcut_alignment * shortcut_strength * (2.0 * clean_y - 1.0) + feature_noise * rng.normal(size=n)
    x[:, 3] = 0.6 * latent[:, 1] + 0.4 * latent[:, 2] + feature_noise * rng.normal(size=n)
    for idx in range(4, input_dim):
        x[:, idx] = feature_noise * rng.normal(size=n)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def make_split_bundle(
    m: int,
    n_v: int,
    n_test: int,
    n_shift: int,
    n_correction: int,
    input_dim: int,
    shortcut_strength: float,
    label_noise: float,
    feature_noise: float,
    seed: int,
    shift_alignment: float = 0.0,
    correction_alignment: float = 0.0,
    correction_eval_alignment: float = 0.0,
) -> SplitBundle:
    rng = np.random.default_rng(seed)
    train = make_shortcut_split(
        n=m,
        input_dim=input_dim,
        shortcut_strength=shortcut_strength,
        shortcut_alignment=1.0,
        label_noise=label_noise,
        feature_noise=feature_noise,
        rng=rng,
    )
    val = make_shortcut_split(
        n=n_v,
        input_dim=input_dim,
        shortcut_strength=shortcut_strength,
        shortcut_alignment=1.0,
        label_noise=label_noise,
        feature_noise=feature_noise,
        rng=rng,
    )
    test_iid = make_shortcut_split(
        n=n_test,
        input_dim=input_dim,
        shortcut_strength=shortcut_strength,
        shortcut_alignment=1.0,
        label_noise=label_noise,
        feature_noise=feature_noise,
        rng=rng,
    )
    test_shift = make_shortcut_split(
        n=n_shift,
        input_dim=input_dim,
        shortcut_strength=shortcut_strength,
        shortcut_alignment=shift_alignment,
        label_noise=label_noise,
        feature_noise=feature_noise,
        rng=rng,
    )
    correction = make_shortcut_split(
        n=n_correction,
        input_dim=input_dim,
        shortcut_strength=shortcut_strength,
        shortcut_alignment=correction_alignment,
        label_noise=label_noise,
        feature_noise=feature_noise,
        rng=rng,
    )
    correction_eval = make_shortcut_split(
        n=max(n_shift, 256),
        input_dim=input_dim,
        shortcut_strength=shortcut_strength,
        shortcut_alignment=correction_eval_alignment,
        label_noise=label_noise,
        feature_noise=feature_noise,
        rng=rng,
    )
    return SplitBundle(
        train=train,
        val=val,
        test_iid=test_iid,
        test_shift=test_shift,
        correction=correction,
        correction_eval=correction_eval,
    )
