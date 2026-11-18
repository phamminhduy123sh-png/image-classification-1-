# Phân loại giống chó mèo bằng PyTorch

EfficientNet-B0 với PyTorch/Torchvision phân loại ảnh thành 37 lớp giống chó mèo của Oxford-IIIT Pet. Giao diện tải ảnh/webcam và biểu đồ kết quả dùng Streamlit.

## Cài trên Windows

PyTorch hỗ trợ Python 3.10–3.14 trên Windows. Trong thư mục project, chạy:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu máy dùng Python khác, đổi `-3.14` thành phiên bản đã cài. Trên CPU, các lệnh trên đủ để bắt đầu. Nếu muốn dùng GPU NVIDIA, lấy lệnh cài PyTorch phù hợp CUDA từ [trang chính thức](https://pytorch.org/get-started/locally/).

## Huấn luyện

```powershell
.\.venv\Scripts\python.exe train.py
```

Lần đầu chương trình tải bộ ảnh Oxford-IIIT Pet và trọng số EfficientNet pretrained. Mặc định huấn luyện đầu phân loại 12 epoch rồi tinh chỉnh hai stage cuối tối đa 5 epoch. Có thể giảm batch size nếu máy thiếu bộ nhớ:

```powershell
.\.venv\Scripts\python.exe train.py --batch-size 16
```

Bỏ qua fine-tuning bằng `--fine-tune-epochs 0`. Kết quả lưu ở `artifacts\model.pth` và `artifacts\metadata.json`.

## Chạy giao diện

Sau khi train xong:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Giao diện có tiêu đề, chọn ảnh/webcam, biểu đồ xác suất 37 giống và thông tin dự đoán.

## Dự đoán một ảnh bằng dòng lệnh

```powershell
.\.venv\Scripts\python.exe predict.py "C:\duong-dan\anh-thu-cung.jpg"
```

Tham khảo: [OxfordIIITPet trong Torchvision](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html), [EfficientNet-B0 và pretrained weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html).
