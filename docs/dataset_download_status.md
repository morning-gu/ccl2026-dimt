# 数据集下载状态报告

## AnyText-benchmark 评测基准

### 当前状态：部分下载（仅元数据，缺少实际图像数据）

已下载的内容（位于 work/datasets/AnyText-benchmark/）：
- AnyText-benchmark.json (244 bytes) - 数据集元数据描述文件，指向 benchmark.zip 和 FID.zip
- benchmark.csv (654 bytes) - 评测集的 caption + 图像路径映射（中英各1000张OCR评测 + 中英各4万张FID评测）
- README.md (1042 bytes) - 数据集说明

**未下载的核心数据文件：**
- benchmark.zip - 包含 OCR 评测所需的图像文件（中英各1000张）
- FID.zip - 包含 FID 评测所需的图像文件（中英各4万张）

### 无法完成下载的原因

1. **ModelScope OSS 存储机制**：benchmark.zip 和 FID.zip 存储在阿里云 OSS 对象存储中，不在 git 仓库内。通过 git clone 只能获取元数据文件。
2. **API 访问受限**：尝试了 ModelScope 的所有已知 API 端点（/repo、/oss、/files、/resolve、/download），均返回 404 或 405。ModelScope 要求使用其 Python SDK（modelscope）的 snapshot_download() 函数来获取文件。
3. **SDK 安装失败**：pip install modelscope 因网络超时无法完成安装。
4. **Google Drive 备用链接**：连接超时。

### 完成下载的方法

**方法一：安装 ModelScope SDK（推荐）**
```
pip install modelscope
python -c "from modelscope.msdatasets import MsDataset; ds = MsDataset.load('iic/AnyText-benchmark', split='test')"
```

**方法二：通过浏览器手动下载**
- 访问 https://modelscope.cn/datasets/iic/AnyText-benchmark/files
- 点击 benchmark.zip 和 FID.zip 的下载按钮
- 或访问 Google Drive: https://drive.google.com/drive/folders/1Eesj6HTqT1kCi6QLyL5j0mL_ELYRp3GV

---

## AnyWord-3M 数据集（300万图像-文本对）

### 当前状态：未下载

### 无法下载的原因

1. **数据集体量巨大**：约 300万张图像，估计总大小约 303GB（160万中文 + 139万英文图像-文本对）。
2. **磁盘空间不足**：当前 C 盘可用空间仅约 18GB，远不够存放完整数据集。
3. **同样依赖 ModelScope SDK**：与 benchmark 相同，需要 modelscope SDK 或浏览器下载。

### 完成下载的方法

**方法一：安装 ModelScope SDK**
```
pip install modelscope
python -c "from modelscope.msdatasets import MsDataset; ds = MsDataset.load('iic/AnyWord-3M', split='train')"
```

**方法二：通过浏览器手动下载**
- 访问 https://modelscope.cn/datasets/iic/AnyWord-3M/files
- 按子目录分批下载各 zip 文件

**部分训练建议**：AnyText 论文提到，使用 20万张图像即可在 8xV100(32GB) 上约 60 小时完成训练，指标虽低于全量但已相当可用。

---

## 总结

| 数据集 | 元数据 | 实际数据 | 主要阻碍 |
|--------|--------|----------|----------|
| AnyText-benchmark | 已下载 | 未下载 | 需 ModelScope SDK 或浏览器 |
| AnyWord-3M | 无 | 未下载 | 303GB，磁盘不足 + 需 SDK |
