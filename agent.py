# agent.py
"""
Agent layer:
- For now, a simple rule-based planner that decides which DBs to use.
- You can later replace `plan_retrieval` with a TinyLlama LLM call.
"""

from dataclasses import dataclass


@dataclass
class RetrievalPlan:
    use_vector: bool
    use_sql: bool
    use_graph: bool
    car_name: str | None = None  # for targeted graph/sql search if possible


def _extract_car_name(question: str) -> str | None:
    """
    Very simple heuristic: look for patterns like 'Car A', 'Car B', etc.
    """
    import re
    matches = re.findall(r"Car\s+[A-Z]", question)
    if matches:
        return matches[0].strip()
    return None

class RetrievalPlan:
    def __init__(self, use_vector=True, use_sql=True, use_graph=False, car_name=None):
        self.use_vector = use_vector
        self.use_sql = use_sql
        self.use_graph = use_graph
        self.car_name = car_name


# Simple routing logic (Easy to replace with TinyLlama later)
def plan_retrieval(question: str) -> RetrievalPlan:
    q = question.lower()

    if "long trip" in q or "best car" in q or "which car" in q:
        return RetrievalPlan(use_sql=True, use_vector=False, use_graph=False)

    return RetrievalPlan()   # default use all DBs


def plan_retrieval(question: str) -> RetrievalPlan:
    """
    Decide which DBs to use based on the question text.
    (Placeholder logic that mimics what a TinyLlama agent would decide.)
    """

    q_lower = question.lower()
    car_name = _extract_car_name(question)

    use_vector = True  # usually always helpful

    use_sql = any(word in q_lower for word in ["trip", "km", "distance", "average", "avg", "usage"])

    use_graph = any(word in q_lower for word in ["feature", "comfort", "relation", "best", "suitable"])

    # fallback: if nothing hits, use all
    if not (use_sql or use_graph):
        use_sql = True
        use_graph = True

    return RetrievalPlan(
        use_vector=use_vector,
        use_sql=use_sql,
        use_graph=use_graph,
        car_name=car_name,
    )
