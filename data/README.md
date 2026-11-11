# 数据集目录

## graphs/
放置图数据 txt 文件（边列表格式）。

每行一条边，格式为 `u v` 或 `u v w`（带权重），以空格/制表符分隔。
`#` 开头的行为注释。

示例（CA-HepTh.txt）：
```
# CA-HepTh 边列表
1 2
2 3 1.0
3 4
```

## labels/
放置标签池 txt 文件，每行一个标签词。

示例（CA-HepTh_labels.txt）：
```
physics
math
quantum
relativity
```

系统会从标签池中随机为每个顶点分配 2-3 个标签。

## 路径配置
在 `../config.py` 的 `DATASETS` 列表中填入你的 txt 文件路径。