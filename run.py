# run.py
from app.rag_query import RAGQuery

def main():
    rag = RAGQuery("index.json")

    while True:
        query = input("\nAsk your question (or 'exit'): ")

        if query.lower() == "exit":
            break

        result = rag.ask(query)
        print("\n--- RAG Answer ---")
        print(result["answer"])
        print("\n--- Context Used ---")
        print(result["context"][:500], "...")
        print("--------------------")

if __name__ == "__main__":
    main()
