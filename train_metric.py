"""Huấn luyện EfficientNet-B0 với Cross-Entropy và batch-hard Triplet Loss."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

CAT_BREEDS = {
    "abyssinian", "bengal", "birman", "bombay", "british shorthair",
    "egyptian mau", "maine coon", "persian", "ragdoll", "russian blue",
    "siamese", "sphynx",
}


class PKBatchSampler(Sampler[list[int]]):
    """Mỗi batch lấy P lớp, mỗi lớp K ảnh để Triplet Loss luôn có positive."""

    def __init__(self, labels, batch_size: int, samples_per_class: int, seed: int):
        if samples_per_class < 2:
            raise ValueError("samples-per-class phải từ 2 trở lên.")
        if batch_size % samples_per_class != 0:
            raise ValueError("batch-size phải chia hết cho samples-per-class.")

        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class
        self.rng = random.Random(seed)
        self.class_indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.class_indices[int(label)].append(index)

        if len(self.class_indices) < self.classes_per_batch:
            raise ValueError(
                f"Cần ít nhất {self.classes_per_batch} lớp trong tập train; "
                f"hiện có {len(self.class_indices)}."
            )

        self.class_ids = list(self.class_indices)
        self.batch_size = batch_size
        self.num_batches = max(1, len(labels) // batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = self.rng.sample(
                self.class_ids, self.classes_per_batch
            )
            batch = []
            for class_id in chosen_classes:
                candidates = self.class_indices[class_id]
                if len(candidates) >= self.samples_per_class:
                    batch.extend(
                        self.rng.sample(candidates, self.samples_per_class)
                    )
                else:
                    batch.extend(
                        self.rng.choices(candidates, k=self.samples_per_class)
                    )
            yield batch


def make_model(
    num_classes: int,
    embedding_dim: int = 128,
    pretrained: bool = True,
) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    # Projection head dùng để học embedding; classifier vẫn giữ cấu trúc cũ.
    model.embedding_head = nn.Linear(in_features, embedding_dim)
    return model


def forward_with_embedding(model: nn.Module, images: torch.Tensor):
    features = model.features(images)
    pooled = model.avgpool(features)
    pooled = torch.flatten(pooled, 1)
    logits = model.classifier(pooled)
    embeddings = F.normalize(model.embedding_head(pooled), p=2, dim=1)
    return logits, embeddings


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Với mỗi ảnh, lấy positive xa nhất và negative gần nhất trong batch."""
    embeddings = embeddings.float()
    distances = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.view(-1, 1)
    same_class = labels.eq(labels.T)
    not_self = ~torch.eye(
        labels.size(0), dtype=torch.bool, device=labels.device
    )
    positive_mask = same_class & not_self
    negative_mask = ~same_class
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)

    # Không có anchor hợp lệ thì trả loss 0 an toàn.
    if not valid.any():
        return embeddings.sum() * 0.0

    hardest_positive = distances.masked_fill(
        ~positive_mask, float("-inf")
    ).max(dim=1).values
    hardest_negative = distances.masked_fill(
        ~negative_mask, float("inf")
    ).min(dim=1).values
    return F.relu(
        hardest_positive[valid] - hardest_negative[valid] + margin
    ).mean()


def configure_trainable_layers(model: nn.Module, fine_tune: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for parameter in model.embedding_head.parameters():
        parameter.requires_grad = True

    if fine_tune:
        for stage in list(model.features.children())[-2:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True


def epoch_pass(
    model,
    loader,
    device,
    triplet_weight: float,
    triplet_margin: float,
    optimizer=None,
    scaler = None,
):
    training = optimizer is not None
    model.train(training)
    # Giữ BatchNorm của backbone cố định như trong code gốc.
    model.features.eval()

    total_loss = total_ce = total_triplet = 0.0
    correct = count = 0
    ce_loss_fn = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits, embeddings = forward_with_embedding(model, images)
                ce_loss = ce_loss_fn(logits, labels)

            triplet_loss = batch_hard_triplet_loss(
                embeddings.float(),
                labels,
                triplet_margin,
            )
            loss = ce_loss + triplet_weight * triplet_loss
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        n = labels.size(0)
        total_loss += loss.item() * n
        total_ce += ce_loss.item() * n
        total_triplet += triplet_loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        count += n

    return (
        total_loss / count,
        correct / count,
        total_ce / count,
        total_triplet / count,
    )


def train_phase(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    title,
    triplet_weight,
    triplet_margin,
):
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=device.type == "cuda",
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2
    )

    best_accuracy = -1.0
    best_state = deepcopy(model.state_dict())
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_ce, train_tri = epoch_pass(
            model, train_loader, device, triplet_weight, triplet_margin,
            optimizer, scaler = scaler
        )
        val_loss, val_acc, val_ce, val_tri = epoch_pass(
            model, val_loader, device, triplet_weight, triplet_margin
        )
        scheduler.step(val_acc)

        print(
            f"{title} {epoch}/{epochs}: "
            f"train loss={train_loss:.4f} (CE={train_ce:.4f}, "
            f"triplet={train_tri:.4f}) acc={train_acc:.2%}; "
            f"val loss={val_loss:.4f} (CE={val_ce:.4f}, "
            f"triplet={val_tri:.4f}) acc={val_acc:.2%}"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                print("Dừng sớm: validation accuracy không cải thiện.")
                break

    model.load_state_dict(best_state)
    return best_state, best_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--triplet-weight", type=float, default=0.3)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.fine_tune_epochs < 0
        or args.batch_size < 1
        or args.samples_per_class < 2
        or args.embedding_dim < 1
        or args.triplet_weight < 0
        or args.triplet_margin <= 0
        or args.num_workers < 0
    ):
        raise SystemExit("Giá trị epochs, batch-size hoặc tham số loss không hợp lệ.")
    if (
        args.batch_size % args.samples_per_class != 0
        or args.batch_size // args.samples_per_class < 2
    ):
        raise SystemExit(
            "batch-size phải chia hết cho samples-per-class và cần ít nhất "
            "2 lớp trong mỗi batch. Ví dụ: --batch-size 32 "
            "--samples-per-class 4."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    weights = EfficientNet_B0_Weights.DEFAULT
    standard = weights.transforms()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.12, contrast=0.12, saturation=0.12
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=standard.mean, std=standard.std),
    ])

    print("Đang tải bộ Oxford-IIIT Pet qua torchvision...")
    train_aug = datasets.OxfordIIITPet(
        args.dataset_dir,
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=True,
    )
    train_eval = datasets.OxfordIIITPet(
        args.dataset_dir,
        split="trainval",
        target_types="category",
        transform=standard,
        download=True,
    )

    class_names = train_aug.classes
    if class_names != train_eval.classes:
        raise RuntimeError("Danh sách nhãn giữa train và validation không khớp.")

    order = torch.randperm(
        len(train_aug),
        generator=torch.Generator().manual_seed(args.seed),
    ).tolist()
    n_val = max(1, round(len(order) * 0.2))
    train_indices = order[n_val:]
    val_indices = order[:n_val]
    train_data = Subset(train_aug, train_indices)
    val_data = Subset(train_eval, val_indices)

    # OxfordIIITPet lưu target số nguyên trong _labels theo thứ tự ảnh.
    train_labels = [int(train_aug._labels[i]) for i in train_indices]
    batch_sampler = PKBatchSampler(
        train_labels,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_data, batch_sampler=batch_sampler, **loader_options
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    model = make_model(
        len(class_names), embedding_dim=args.embedding_dim, pretrained=True
    ).to(device)
    configure_trainable_layers(model, fine_tune=False)

    best_state, best_val = train_phase(
        model, train_loader, val_loader, device, args.epochs, 1e-3,
        "Đầu phân loại + metric", args.triplet_weight, args.triplet_margin,
    )

    if args.fine_tune_epochs:
        model.load_state_dict(best_state)
        configure_trainable_layers(model, fine_tune=True)
        tuned_state, tuned_val = train_phase(
            model, train_loader, val_loader, device,
            args.fine_tune_epochs, 1e-5, "Fine-tune + metric",
            args.triplet_weight, args.triplet_margin,
        )
        if tuned_val >= best_val:
            best_state = tuned_state
            best_val = tuned_val
        model.load_state_dict(best_state)

    species = {
        name: (
            "mèo"
            if name.lower().replace("_", " ") in CAT_BREEDS
            else "chó"
        )
        for name in class_names
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_state = model.state_dict()
    # state_dict phân loại giữ nguyên key để common.py/predict.py hiện tại
    # vẫn nạp được; metric head được lưu riêng để dùng khi cần embedding.
    classifier_state = {
        key: value.detach().cpu()
        for key, value in full_state.items()
        if not key.startswith("embedding_head.")
    }
    metric_head_state = {
        key: value.detach().cpu()
        for key, value in model.embedding_head.state_dict().items()
    }
    checkpoint = {
        "state_dict": classifier_state,
        "metric_head_state_dict": metric_head_state,
        "class_names": class_names,
        "species_by_breed": species,
        "image_size": 224,
        "validation_accuracy": best_val,
        "metric_learning": "batch-hard triplet loss",
        "embedding_dim": args.embedding_dim,
        "triplet_weight": args.triplet_weight,
        "triplet_margin": args.triplet_margin,
        "samples_per_class": args.samples_per_class,
    }
    torch.save(checkpoint, args.output_dir / "model.pth")

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"state_dict", "metric_head_state_dict"}
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Đã lưu model: {args.output_dir / 'model.pth'}")
    print(f"Validation accuracy tốt nhất: {best_val:.2%}")
    print("Checkpoint có metric_head_state_dict để dùng embedding về sau.")


if __name__ == "__main__":
    main()
