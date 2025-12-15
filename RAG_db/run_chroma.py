from app.rag_query_chroma import RAGQueryChroma

rag = RAGQueryChroma()

while True:
    q = input("\nAsk (or exit): ")
    if q.lower() == "exit":
        break

    answer = rag.ask(q)

    print("\n--- Answer ---")
    print(answer)
