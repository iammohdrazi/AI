from app.rag_query_faiss import RAGQueryFAISS


def main():
    rag = RAGQueryFAISS()

    while True:
        query = input("\nAsk your question (or 'exit'): ")

        if query.lower() == "exit":
            break

        result = rag.ask(query)

        print("\n--- Answer ---")
        print(result["answer"])
        print("\n--- Context Used ---")
        print(result["context"][:500], "...")
        print("--------------------")


if __name__ == "__main__":
    main()
