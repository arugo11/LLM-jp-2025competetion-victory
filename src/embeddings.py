from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_text_splitter(chunk_size=100, chunk_overlap=20):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )


def get_embedding_provider(model_name: str = "intfloat/multilingual-e5-small"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


class EmbeddingModel:
    def __init__(self, text_splitter=None, embedding_provider=None):
        if text_splitter is None:
            text_splitter = get_text_splitter()
        if embedding_provider is None:
            embedding_provider = get_embedding_provider()

        self.text_splitter = text_splitter
        self.embedding_provider = embedding_provider

    def create_vectorstore(self, text: str):
        # Split the text into chunks
        texts = self.text_splitter.split_text(text)
        # Create vectorstore
        vectorstore = FAISS.from_texts(texts, self.embedding_provider)

        return vectorstore

    def create_vectorstore_from_qa(self, qa_pairs: list[dict]):
        """questionをベクトル化し、answerをmetadataとして保存"""
        questions = [qa["problem"] for qa in qa_pairs]
        metadatas = [{"generated_solution": qa["generated_solution"]} for qa in qa_pairs]
        vectorstore = FAISS.from_texts(
            questions, self.embedding_provider, metadatas=metadatas
        )
        return vectorstore


class Retriever:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.retriever = self.get_retriever()

    def save(self, file_path: str):
        self.vectorstore.save_local(file_path)

    @staticmethod
    def load(file_path: str, embedding_provider=None):
        if embedding_provider is None:
            embedding_provider = get_embedding_provider()

        vectorstore = FAISS.load_local(
            file_path, embedding_provider, allow_dangerous_deserialization=True
        )
        return Retriever(vectorstore)

    def get_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        return retriever

    def invoke_content(self, query: str) -> list[str]:
        documents = self.retriever.invoke(query)
        contents = [doc.page_content for doc in documents]
        return contents

    def invoke_answers(self, query: str) -> list[dict]:
        """クエリに対してquestionとanswerのペアを返す"""
        documents = self.retriever.invoke(query)
        results = []
        for doc in documents:
            results.append({
                "question": doc.page_content,
                "answer": doc.metadata.get("generated_solution", ""),
            })
        return results
