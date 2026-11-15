from typing import List
from agent import chat

import chromadb
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer


# 切分知识库
def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r', encoding="UTF-8") as file:
        content = file.read()

    return [chunk for chunk in content.split("\n\n")]


def embed_chunk(chunk: str) -> List[float]:
    # 知识库向量化
    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


def save_embeddings(chunks: List[str], embeddings: List[List[float]], chromadb_collection) -> None:
    # 知识库存储
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )


# 知识库召回
def retrieve(query: str, top_k: int, chromadb_collection=None) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]


# 知识库重排
def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


def main():
    # 切分知识库
    chunks = split_into_chunks("doc.md")
    for i, chunk in enumerate(chunks):
        print(f"{i}, {chunk}")

    print()
    print("#" * 150)
    print()

    # 知识库向量化
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    print()
    print("#" * 150)
    print()

    # 保存知识库
    chromadb_client = chromadb.EphemeralClient()
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")
    save_embeddings(chunks, embeddings, chromadb_collection)
    print()
    print("#" * 150)
    print()

    # 知识库召回
    query = "哆啦A梦使用的3个秘密道具分别是什么？"
    retrieved_chunks = retrieve(query, 5, chromadb_collection)
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk}\n")
    print()
    print("#" * 150)
    print()

    # 知识库重排
    reranked_chunks = rerank(query, retrieved_chunks, 3)
    for i, chunk in enumerate(reranked_chunks):
        print(f"[{i}] {chunk}\n")

    user_input = "哆啦A梦的三个密码道具是什么？"
    res = chat(user_input, reranked_chunks)
    print(res)


if __name__ == "__main__":
    main()
