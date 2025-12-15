from app.rag_query_faiss import RAGQueryFAISS
from app.utils.spinner import Spinner


def main():
    rag = RAGQueryFAISS()

    while True:
        query = input("\nAsk (or exit): ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            break

        spinner = Spinner("Finding answer")
        spinner.start()

        try:
            result = rag.ask(query)
        finally:
            spinner.stop()

        print("\n--- Answer ---")
        print(result.get("answer", "No answer generated"))


if __name__ == "__main__":
    main()
