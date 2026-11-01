# SmartDoc 50 张原图的来源与真实四角

本目录是程序评价所需的元数据。`dataset` 中的原始图像未被修改。

- 数据集：SmartDoc 2015 Challenge 1，v2.0.0 官方帧图像重打包版。
- 作者：Joseph Chazalon、Marçal Rusiñol。
- 原始数据记录：https://doi.org/10.5281/zenodo.1230218
- 发布仓库：https://github.com/jchazalon/smartdoc15-ch1-dataset
- 许可：CC BY 4.0，随附发布者许可文件；课程报告和再分发应保留作者署名和来源。
- 发布包 SHA-256：`3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae`。

`smartdoc50_annotations.json` 提供 `doc_001.jpg` 到 `doc_050.jpg` 的原始路径、SHA-256、四角与文档类别。
所有 50 张本地 JPEG 均已与官方归档中的源 JPEG 做 SHA-256 一致性比较；这些坐标是发布者真实标注，而不是检测算法输出。
四角顺序：左上、右上、右下、左下。坐标以原始 1920×1080 图像为准，缩放时须同步缩放坐标。

选择规则：5 背景各 10 张；每序列最多 1 帧；覆盖全部 30 文档、6 类。按文档名称排序，背景索引 b=0..4 选模型下标 range(b%3,30,3)。
候选帧四角距边至少 12 像素、面积至少占图像 8%；按帧号排序，选取分位点 .25+.1*(序列局部索引%6)，位置使用 round((n-1)*q)。
仅重命名，保留发布者原始 JPEG 字节与拍摄缺陷。此子集不是完整官方 benchmark。

恢复日期：2026-11-01。
