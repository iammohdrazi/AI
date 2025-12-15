from app.rag_query_faiss import RAGQueryFAISS

rag = RAGQueryFAISS()

while True:
    q = input("\nAsk (or exit): ")
    if q.lower() == "exit":
        break

    res = rag.ask(q)
    print("\n--- Answer ---")
    print(res["answer"])
