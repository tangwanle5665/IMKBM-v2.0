# 智能医学知识库 （基于RAG模型的考研医学助手）

## 引言

本医学知识库RAG模型旨在帮助广大医学生解决考研过程中，存在的知识点过于分散，书本过厚，考研卷子难度日益增加且对于医学知识整体框架的构建的要求日益增加的问题

经过实验，对于向量库进行编码后，对于不同书本内类似知识点的潜在逻辑可以得到快速且有效的串联，有助于使用者对于跨书本的医学知识进行深刻的理解

------

## 项目中使用的插件以及对应插件的功能介绍：
1. numpy：在该知识库中，主要用来进行所有向量和矩阵的计算
2. scipy：主要负责稀疏这个项目中的矩阵，从而起到节省内存的作用
3. scikit-learn：主要负责在文本向量化，检索过程中所需要的向量余弦角度计算以及归一化
4. pandas：主要负责读取题库为表格数据
5. fastembed：这个项目中的主力，主要负责加载官方 ONNX 版 BGE 模型，当前实际使用的低内存嵌入引擎，可以降低对于电脑配置的要求
6. onnxruntime：fastembed的推理引擎，不依赖pytorch的推理引擎，可以减少内存要求
7. sentence-transformers：备用的嵌入方案（PyTorch 版），fastembed 不可用时回退，增加系统运行的稳定性
8. torch：日常运行不加载，只在pytorch版本运行的时候他才运行
9. transformers / tokenizers：sentence-transformers 底层依赖
10. huggingface-hub：可以从从HF镜像网站里下载模型文件
11. faiss-cpu：可以进行向量相似度索引，也是该系统运行过程中的核心
12. PyMuPDF (fitz)：负责解析 PDF（教材、考研真题、药品说明书）提取文字
13. python-docx：负责读取docx教材/题库+生成复习册/试卷 docx
14. python-pptx负责解析教学PPT（.ppt/.pptx）提取文字
15. comtypes负责调用 Word COM 把老式 .doc 转成 .docx（更新题库用）
16. langchain-text-splitters：负责将教材长文本按500字智能分块（构建向量库的 65,929 个块）
17. langchain-core：text-splitters 的底层依赖（Document 数据结构）
18. matplotlib：负责绘制代谢路径图（三羧酸循环、糖酵解、尿素循环等 PNG）
19. networkx：负责代谢通路图结构布局辅助
20. Pillow：负责GUI/问答中图片的缩放、预览、显示
21. flask：负责手机版/公网版Web服务
22. requests：负责调用DeepSeek API
23. python-dotenv：读取env中的环境变量（config.py加载配置）
24. tqdm：负责显示向量编码过程中的进度条，不然这会是一个很枯燥的过程，你永远都不知道你还要等多久
25. joblib：计算还剩多久嗯对

numpy，scipy，scikit-learn，pandas为科学计算与数据处理所需要的插件
onnxruntime，fastembed，sentence-transformers，torch，transformers / tokenizers，huggingface-hub都是AI嵌入模型，是语义检索的核心
faiss-cpu负责对向量进行检索
PyMuPDF (fitz)，python-docx，python-pptx，comtypes负责进行文档的解析（包括题库，药物说明书，教材等）
langchain-text-splitters，langchain-core则负责进行文本分块
matplotlib，networkx，Pillow负责绘制可视化图像
针对网页版，我选取了flask，requests，python-dotenv这三个插件进行运作
向量编码的过程是一个十分折磨人的过程，为了然用户更直观的感受还剩多久，我选取了tqdm和joblib来绘制进度条


---
## 相较于上一个版本的改变
1. 该医学知识库基于模块RAG，方便后续对于其功能进行完善，受制于电脑性能的限制，无法使用图表RAG

---
## 对于电脑配置的要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows 10/11 64位 | 32 位不行 |
| Python | miniconda 自带（当前 3.14） | 一键下载安装.bat 自动装 |
| 依赖包 | torch / faiss / fastembed / sentence-transformers / flask 等 20+ 个 | 自动安装（阿里云镜像） |
| AI 模型 | BGE fastembed v1.5（已打包在 models/） | 离线可用，不联网也能检索 |
| 网络 | ✅ 必须联网（DeepSeek API 生成回答 更改代码可以改成其他模型） | 问答要调云端大模型 |
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
3. 若想拜托局域网的束缚，则可以点击公网版.bat，点击这个bat文件会弹出两个cmd窗口，在较杂乱的日志中找到类似于https://44538ef8.r39.cpolar.top 的结构，复制粘贴或直接ctrl并点击即可进入到对应的网页
4. 电脑可能会存在定期休眠的情况，用户可以根据自己的需求来决定是否需要选择24h启动.bat
5. 用户在使用过程中如果改变嵌入模型，例如我将之前的PyTorch bge-small-zh（384 维）转变为fastembed bge-small-zh-v1.5（512 维）则需要重新运行重建向量库.bat
6. 内部插件版本可能存在不定期更新的情况，若是无法运行，则可以优先试试看一件维护.bat，若是还是无法运行，我的手机号为19703675390，我们可以在微信上交流
7. 输出的填空题就在这个目录下面，用户若是想要出一份检测自己知识掌握程度的试卷，则需要定期将数据库文件“复习填空题_data.json”复制到这个目录：D/python work/医学知识库/考试题库/数据库 这个目录下面，出出来的试卷则放在D/python work/医学知识库/考试题库/试卷 这个目录下面

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
