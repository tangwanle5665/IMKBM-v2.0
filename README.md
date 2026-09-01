# 医学知识库 RAG

## 引言

本医学知识库RAG模型旨在帮助广大医学生解决考研过程中，存在的知识点过于分散，书本过厚，考研卷子难度日益增加且对于医学知识整体框架的构建的要求日益增加的问题

经过实验，对于向量库进行编码后，对于不同书本内类似知识点的潜在逻辑可以得到快速且有效的串联，有助于使用者对于跨书本的医学知识进行深刻的理解

---

## 对于电脑配置的要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows 10/11 64位 | 32 位不行 |
| Python | miniconda 自带（当前 3.14） | 一键下载安装.bat 自动装 |
| 依赖包 | torch / faiss / fastembed / sentence-transformers / flask 等 20+ 个 | 自动安装（阿里云镜像） |
| AI 模型 | BGE fastembed v1.5（已打包在 models/） | 离线可用，不联网也能检索 |
| 网络 | ✅ 必须联网（DeepSeek API 生成回答） | 问答要调云端大模型 |
| API Key | DeepSeek（已在 config.py 内置） | 无需手动配置 |
| 公网（可选） | cpolar（隧道工具） | 只在用"公网版"时需要 |

### 三个容易忽略的点

1. **问答必须联网**：检索是本地（离线），但生成回答靠 DeepSeek 云端 API——断网时能检索不能回答
2. **杀毒软件**：360 等可能拦截 python/pip/安装器，安装和运行时建议关闭或加白名单
3. **防火墙**：手机版/公网版需要放行 5000 端口（脚本已自动加规则，首次运行弹窗点"允许"）

---

## 安装过程

1. 在D盘创建路径，为D/python work/医学知识库
2. 将17个py文件源代码放入到这个文件夹当中
3. 创立pdf，ppt，drug_labels，汤臣一品等文件夹
4. 将目录里所有的bat文件放入到汤臣一品当中
5. 点击一键安装.bat，就可以布置好知识库运行所需要的全部环境
6. 输入win+R，进入cmd界面，输入cpolar authtoken 你的Token（Token 在 cpolar 官网 → 验证 → Authtoken 页面查看，首次使用需要申请）

---
## 运行过程
1. 若是想直接点击桌面版，可以进入项目目录，点击AAA建材王总.bat
2. 想同一个局域网下面所有的用户都可以使用，则可以点击手机版.bat
3. 若想拜托局域网的束缚，则可以点击公网版.bat，点击这个bat文件会弹出两个cmd窗口，在较杂乱的日志中找到类似于https://44538ef8.r39.cpolar.top的结构，复制粘贴或直接ctrl并点击即可进入到对应的网页
---

## 项目结构

```
医学知识库/
├── 核心代码（17 个 py 文件）
│   ├── med_qa.py          ← 问答主逻辑（RAG+填空+标签）
│   ├── gui_app.py         ← 桌面版界面
│   ├── web_app.py         ← 手机/公网版服务
│   ├── utils.py           ← 工具（fastembed/检索/扩展词）
│   ├── retrieval_test.py  ← 混合检索（TF-IDF+FAISS SQ8）
│   ├── build_vector_db.py ← 向量库构建
│   ├── question_bank.py   ← 题库加载+相似题
│   ├── exam_paper.py      ← 试卷生成（40/40/20）
│   ├── exam_aligner.py    ← 真题导入对齐
│   ├── update_question_bank.py ← 真题解析
│   ├── full_update.py     ← 增量更新
│   ├── pdf/ppt_parser.py  ← 教材解析
│   ├── drug_label_parser.py ← 说明书解析
│   ├── download_model.py  ← 模型下载
│   └── config.py          ← 全局配置
│
├── rag_modules/           ← 流水线模块（pipeline/router/memory/security）
├── vector_db/ 558MB       ← 向量库（fastembed v1.5）
├── models/ 365MB          ← AI模型（bge + fastembed ONNX）
├── parsed_texts/ 66MB     ← 教材解析文本
├── pdf                    ← 教材源文件
├── ppt                    ← 教学PPT
├── drug_labels            ← 药物说明书
├── 题库pdf/               ← 考研真题
├── 题库doc/               ← Word题库
├── 考试题库/              ← 试卷+数据库
├── question_bank/         ← 题库向量缓存
├── images/                ← 代谢路径图
├── 汤臣一品/               ← 全部脚本（7 bat + 重建工具）
├── saved_sessions/        ← GUI会话历史
├── 数据文件：题库*.csv、复习填空题_data.json/docx、knowledge_history.json、answer_cache.json
└── 项目结构.txt           ← 本文件
```
