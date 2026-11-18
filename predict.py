"""Dự đoán giống chó/mèo từ một ảnh bằng mô hình PyTorch."""
import argparse
from pathlib import Path

from PIL import Image

from common import load_classifier, predict_top_k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Đường dẫn tới ảnh.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Không tìm thấy ảnh: {args.image}")
    model, metadata, device, transform = load_classifier(args.artifacts)
    with Image.open(args.image) as image:
        results = predict_top_k(model, metadata, device, transform, image, args.top_k)
    print(f"Ảnh: {args.image} | thiết bị: {device}")
    for rank, item in enumerate(results, 1):
        print(f"{rank}. Giống {item['species']} {item['breed']}: {item['confidence']:.2%}")


if __name__ == "__main__":
    main()
