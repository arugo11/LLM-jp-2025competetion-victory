import os

import embeddings
import utils

parquet_directory = "data"
vectorstore_path = "data/qa_vectorstore"


def main():
    if os.path.exists(vectorstore_path):
        print("Loading existing vectorstore...")
        retriever = embeddings.Retriever.load(vectorstore_path)
        print("Loaded existing vectorstore.")
    else:
        print("Loading QA data from parquet files...")
        qa_pairs, loaded_files = utils.read_parquet_qa_from_directory(parquet_directory)
        print(f"Loaded {len(loaded_files)} parquet file(s):")
        for f in loaded_files:
            print(f"  - {f}")
        print(f"Total {len(qa_pairs)} QA pairs.")

        print("Creating vectorstore...")
        embedding_model = embeddings.EmbeddingModel()
        vectorstore = embedding_model.create_vectorstore_from_qa(qa_pairs)
        retriever = embeddings.Retriever(vectorstore)
        retriever.save(vectorstore_path)
        print("Vectorstore created and saved.")

    while True:
        query = input("\nEnter a query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = retriever.invoke_answers(query)
        print("\nResults:")
        for i, result in enumerate(results):
            print("=" * 20 + f" Result {i + 1} " + "=" * 20)
            print(f"Q: {result['question'][:100]}...")
            print(f"A: {result['answer']}")


if __name__ == "__main__":
    main()
