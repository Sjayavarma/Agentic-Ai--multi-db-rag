# ==========================================================
#  app.py  🚀 Fully Working Multi-DB Agentic RAG + Gemma-2B
# ==========================================================

from agent import plan_retrieval
from dbs import vector_search, sql_query_by_car_keyword, graph_query_by_car
from gemma_generator import generate_final_answer


DEBUG_CONTEXT = True   # Set False to hide context in console


# ---------------------------------------------------------
# Build prompt context → fed into Gemma for reasoning
# ---------------------------------------------------------
def build_context(question: str, vector_docs, sql_rows, graph_rows) -> str:
    parts = [f"User Question: {question}\n"]

    if sql_rows:
        parts.append("\n[SQL RESULTS - Car Stats]\n")
        for row in sql_rows:
            parts.append(f"- {row}\n")

    if vector_docs:
        parts.append("\n[VECTOR MATCHED DOCS - Descriptions]\n")
        for d in vector_docs:
            parts.append(f"- {d.get('text')}\n")

    if graph_rows:
        parts.append("\n[GRAPH RELATIONS - Car → Feature]\n")
        for g in graph_rows:
            parts.append(f"- {g.get('car')} --{g.get('relation')}--> {g.get('feature')}\n")

    return "".join(parts)


# ---------------------------------------------------------
# (Long Trip Filter Logic)
# ---------------------------------------------------------
def pick_top_long_trip_cars(sql_rows, top_n: int = 8):
    scored = []
    for row in sql_rows:
        try:
            avg_km = float(row.get("avg_km", 0))
            trips = float(row.get("trips", 0))
        except ValueError:
            continue

        score = avg_km * 0.7 + trips * 10
        row_copy = dict(row)
        row_copy["long_trip_score"] = round(score, 2)
        scored.append((score, row_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_n]]


# ---------------------------------------------------------
# MAIN ANSWER ENGINE
# ---------------------------------------------------------
def answer_user(question: str) -> str:

    # 1) Agent routing (LLM or rule-based for now)
    plan = plan_retrieval(question)

    # 2) DB access based on agent decision
    vector_docs = vector_search(question) if plan.use_vector else []
    car_kw = plan.car_name if plan.car_name else "Car"
    sql_raw = sql_query_by_car_keyword(car_kw) if plan.use_sql else []
    graph_rows = graph_query_by_car(car_kw) if plan.use_graph else []

    # 3) FILTER SQL results → make reasoning effective
    sql_rows = pick_top_long_trip_cars(sql_raw, top_n=10)

    # 4) Build final RAG context
    context = build_context(question, vector_docs, sql_rows, graph_rows)

    if DEBUG_CONTEXT:
        print("\n========== 🔍 CONTEXT SENT TO GEMMA ==========\n")
        print(context)
        print("\n==============================================\n")

    return generate_final_answer(question, context)


# ---------------------------------------------------------
# APPLICATION LOOP
# ---------------------------------------------------------
def main():
    print("\n=== Agentic Multi-DB RAG Demo (FAISS + CSV + Gemma-2B) ===")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit", "bye"):
            print("\n🔚 Session Ended. Goodbye!")
            break

        print("\nAssistant:\n", answer_user(q))


if __name__ == "__main__":
    main()
