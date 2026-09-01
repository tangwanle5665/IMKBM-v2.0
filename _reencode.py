# -*- coding: utf-8 -*-
"""重建向量库 BGE 部分（fastembed v1.5, 512维）——单进程流式版（最省内存）
逐批编码 + memmap 直接写盘，内存占用最小，适合低配置电脑
只重编码 bge_vectors，TF-IDF/chunks 复用现有 hybrid_index.pkl"""
import os, sys, io, pickle, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
BASE = r"D:\python work\医学知识库"
VDB = os.path.join(BASE, "vector_db")
PKL = os.path.join(VDB, "hybrid_index.pkl")
NPY = os.path.join(VDB, "bge_vectors.npy")
TMP_NPY = os.path.join(VDB, "bge_vectors.npy.tmp")
CACHE = os.path.join(BASE, "models", "fastembed_cache")
MODEL = "BAAI/bge-small-zh-v1.5"
BATCH = 128   # 每批编码条数（内存紧张可改 64）


def main():
    t0 = time.time()
    print("加载现有索引...")
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    texts = [c.page_content for c in chunks]
    n = len(texts)
    # 立即释放旧索引（旧向量库等大对象），只保留文本，降低峰值内存
    del data, chunks
    import gc
    gc.collect()
    print(f"共 {n} 块，单进程流式编码（fastembed v1.5）")

    # 加载嵌入模型（单个实例，不复制）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from fastembed import TextEmbedding
    emb = TextEmbedding(model_name=MODEL, cache_dir=CACHE, threads=2)

    # memmap 直接写盘：向量不占内存
    dim = 512
    bge = np.memmap(TMP_NPY, dtype=np.float32, mode="w+", shape=(n, dim))

    from tqdm import tqdm
    pbar = tqdm(total=n, desc="BGE编码 v1.5", unit="块", ncols=90)
    done = 0
    for i in range(0, n, BATCH):
        batch = texts[i:i + BATCH]
        vecs = np.array(list(emb.embed(batch, batch_size=BATCH)), dtype=np.float32)
        bge[done:done + vecs.shape[0]] = vecs
        done += vecs.shape[0]
        pbar.update(vecs.shape[0])
        del vecs
    pbar.close()
    bge.flush()
    del bge

    # 落盘并更新索引
    os.replace(TMP_NPY, NPY)
    data["bge_vectors"] = np.load(NPY, mmap_mode="r")
    with open(PKL, "wb") as f:
        pickle.dump(data, f)
    print(f"完成: bge_vectors 已更新为 {data['bge_vectors'].shape}（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
