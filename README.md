# Transformer

## 环境

### 安装uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 同步依赖

```bash
uv sync
```

### 下载数据集

```bash
make data
```


## 训练

```bash
uv run train.py
```

## 结果

`saved/model-best.pt` 就是训练的产物
