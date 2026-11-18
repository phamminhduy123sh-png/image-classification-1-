import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from common import load_classifier


def evaluate_retrieval(embeddings, labels, margin=0.3, seed=42):
    """
    Chia test thành gallery/query theo từng lớp, rồi tính:
    - Recall@1, Recall@5
    - batch-hard triplet loss và triplet violation rate

    embeddings phải được L2-normalize trước khi truyền vào.
    """
    class_indices = defaultdict(list)
    for index, label in enumerate(labels):
        class_indices[int(label)].append(index)

    gallery_indices = []
    query_indices = []

    # Chia cố định theo từng lớp để có cùng gallery/query khi chạy lại.
    for label in sorted(class_indices):
        indices = class_indices[label]

        if len(indices) < 2:
            continue

        rng = random.Random(seed + label)
        indices = indices.copy()
        rng.shuffle(indices)

        split_point = len(indices) // 2
        gallery_indices.extend(indices[:split_point])
        query_indices.extend(indices[split_point:])

    if not gallery_indices or not query_indices:
        raise RuntimeError(
            "Không đủ ảnh để chia gallery/query. "
            "Cần ít nhất 2 ảnh cho mỗi giống trong tập test."
        )

    gallery_idx = torch.tensor(
        gallery_indices, dtype=torch.long, device=embeddings.device
    )
    query_idx = torch.tensor(
        query_indices, dtype=torch.long, device=embeddings.device
    )

    gallery_embeddings = embeddings[gallery_idx]
    query_embeddings = embeddings[query_idx]

    all_labels = torch.tensor(
        labels, dtype=torch.long, device=embeddings.device
    )
    gallery_labels = all_labels[gallery_idx]
    query_labels = all_labels[query_idx]

    # Vì embedding đã normalize, tích vô hướng là cosine similarity.
    similarities = query_embeddings @ gallery_embeddings.T
    max_k = min(5, len(gallery_indices))
    nearest_indices = similarities.topk(max_k, dim=1).indices
    nearest_labels = gallery_labels[nearest_indices]

    recall_at_1 = (
        nearest_labels[:, :1]
        .eq(query_labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    recall_at_5 = (
        nearest_labels[:, :max_k]
        .eq(query_labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    # Tính hardest positive và hardest negative cho mỗi ảnh query.
    distances = torch.cdist(
        query_embeddings.float(),
        gallery_embeddings.float(),
        p=2,
    )

    same_class = query_labels.unsqueeze(1).eq(gallery_labels.unsqueeze(0))
    has_positive = same_class.any(dim=1)
    has_negative = (~same_class).any(dim=1)
    valid = has_positive & has_negative

    if not valid.any():
        raise RuntimeError("Không tạo được triplet hợp lệ từ gallery/query.")

    hardest_positive = distances.masked_fill(
        ~same_class,
        float("-inf"),
    ).max(dim=1).values

    hardest_negative = distances.masked_fill(
        same_class,
        float("inf"),
    ).min(dim=1).values

    triplet_losses = F.relu(
        hardest_positive[valid] - hardest_negative[valid] + margin
    )

    violation_rate = (
        hardest_positive[valid] + margin >= hardest_negative[valid]
    ).float().mean().item()

    return {
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "test_triplet_loss": triplet_losses.mean().item(),
        "triplet_violation_rate": violation_rate,
        "retrieval_query_count": len(query_indices),
        "retrieval_gallery_count": len(gallery_indices),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Thư mục chứa model.pth, metadata.json và nơi lưu kết quả.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, metadata, device, transform = load_classifier(args.artifacts)
    class_names = metadata["class_names"]
    labels = list(range(len(class_names)))

    artifacts_dir = Path(args.artifacts)
    checkpoint_path = artifacts_dir / "model.pth"

    # Checkpoint triplet có metric_head_state_dict riêng.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    metric_head_state = checkpoint.get("metric_head_state_dict")
    metric_head = None

    if metric_head_state is not None:
        in_features = model.classifier[1].in_features
        embedding_dim = int(
            checkpoint.get(
                "embedding_dim",
                metric_head_state["weight"].shape[0],
            )
        )

        metric_head = nn.Linear(in_features, embedding_dim).to(device)
        metric_head.load_state_dict(metric_head_state)
        metric_head.eval()

    # Dùng split test riêng để đánh giá.
    test_data = OxfordIIITPet(
        root=args.dataset_dir,
        split="test",
        target_types="category",
        transform=transform,
        download=True,
    )

    if test_data.classes != class_names:
        raise RuntimeError(
            "Nhãn của tập test không khớp thứ tự class_names trong checkpoint."
        )

    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    y_true = []
    y_pred = []
    embedding_batches = []

    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )

            # Trích xuất EfficientNet features một lần.
            features = model.features(images)
            pooled = model.avgpool(features)
            pooled = torch.flatten(pooled, 1)

            logits = model.classifier(pooled)
            predictions = logits.argmax(dim=1)

            if metric_head is not None:
                # Model Triplet: dùng embedding head đã học.
                embeddings = metric_head(pooled)
                embedding_source = "metric_head"
            else:
                # Model Cross-Entropy cũ: dùng vector trước classifier.
                embeddings = pooled
                embedding_source = "backbone_features"

            embeddings = F.normalize(embeddings.float(), p=2, dim=1)

            y_true.extend(targets.tolist())
            y_pred.extend(predictions.cpu().tolist())
            embedding_batches.append(embeddings)

    all_embeddings = torch.cat(embedding_batches, dim=0)

    # Chỉ số phân loại.
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"Thiết bị: {device}")
    print(f"Loại embedding: {embedding_source}")
    print(report_text)

    # Chỉ số metric learning / image retrieval.
    retrieval = evaluate_retrieval(
        all_embeddings,
        y_true,
        margin=args.triplet_margin,
        seed=args.seed,
    )

    print("\nĐánh giá embedding:")
    print(f"Recall@1: {retrieval['recall_at_1']:.4f}")
    print(f"Recall@5: {retrieval['recall_at_5']:.4f}")
    print(f"Test triplet loss: {retrieval['test_triplet_loss']:.4f}")
    print(
        "Triplet violation rate: "
        f"{retrieval['triplet_violation_rate']:.2%}"
    )

    output_dir = artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "embedding_source": embedding_source,
        "triplet_margin_used_for_evaluation": args.triplet_margin,
        **retrieval,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Hàng là giống thật, cột là giống model dự đoán.
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with (output_dir / "confusion_matrix.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Thực tế / Dự đoán", *class_names])

        for name, row in zip(class_names, cm.tolist()):
            writer.writerow([name, *row])

    # Ma trận chuẩn hóa theo từng giống thật.
    cm_normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(18, 16))
    ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=class_names,
    ).plot(
        include_values=False,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"\nĐã lưu kết quả trong: {output_dir}")


if __name__ == "__main__":
    main()