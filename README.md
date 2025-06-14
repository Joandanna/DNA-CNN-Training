# DNA-CNN-Training
# README - 启动子识别训练脚本说明

本项目用于使用卷积神经网络（CNN）模型识别大肠杆菌的启动子序列，采用 PyTorch 框架实现。数据源于 RegulonDB 数据库中的正例和负例 fasta 序列文件。

## 📁 项目目录结构

```
├── train.py                  # 主训练脚本
├── simple_dna_cnn.pt         # 训练好的模型权重文件（保存路径）
├── /data/
│   ├── promoter_train_pos.fasta   # 启动子正例训练集
│   ├── promoter_train_neg.fasta   # 启动子负例训练集
│   ├── promoter_valid_pos.fasta   # 启动子正例验证集
│   ├── promoter_valid_neg.fasta   # 启动子负例验证集
```

## 🧪 脚本功能说明

### `load_fasta_as_dataset(path, label)`

从 fasta 文件中读取 DNA 序列并进行 one-hot 编码，返回编码后的序列张量和对应标签列表。

* 参数：

  * `path`: FASTA 文件路径
  * `label`: 该文件对应的分类标签（1: 正例，0: 负例）

### `file_label_map`

定义每个文件路径及其标签，按如下规则：

```python
file_label_map = {
  'promoter_train_pos.fasta': 1,
  'promoter_train_neg.fasta': 0,
  'promoter_valid_pos.fasta': 1,
  'promoter_valid_neg.fasta': 0
}
```

### 加载与拼接数据：

将所有 fasta 文件读取并合并为训练数据 `X_all`、标签 `y_all`。
脚本中会自动检查文件是否存在，缺失则跳过，并提示 `文件未找到`。

### Debug 注意事项：

运行训练前，请确保：

* 所有 `.fasta` 文件存在且格式正确（包含 `>header` 行）
* `load_fasta_as_dataset` 返回的 `sequences` 非空

你可通过如下命令检查：

```python
print("训练样本总数:", len(X_all))
```

## 🚀 启动训练方法

```bash
python train.py
```

或使用 Jupyter Notebook 执行相关 cell。

## 🧠 模型结构概述（CNN）

* 输入张量：大小为 `[batch_size, 4, 81]`（4个通道表示 A/C/G/T）
* Conv1D 层：提取序列局部模式
* MaxPool 层：降低维度，增强特征鲁棒性
* Fully Connected：Sigmoid 激活进行二分类

## ⚠️ 常见错误说明

* `ValueError: num_samples=0`

  * 说明加载的数据为空。
  * 请检查 fasta 文件路径、内容格式、是否成功读取。

## ✅ 示例运行日志（前几轮训练）

```
Epoch 1, Loss: 0.6907
Epoch 2, Loss: 0.5963
Epoch 3, Loss: 0.7029
Epoch 4, Loss: 0.4004
...
Epoch 9, Loss: 0.4492
```

## 📬 联系方式

如需调试帮助或模型可视化扩展，请联系作者：3022210113@tju.edu.cn
