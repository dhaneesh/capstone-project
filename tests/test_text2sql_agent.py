import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text2sql_agent import create_initial_state, sanitize_sql_response


def test_create_initial_state_populates_defaults():
    state = create_initial_state("How many orders were delivered?")

    assert state["question"] == "How many orders were delivered?"
    assert state["sql_query"] == ""
    assert state["query_result"] == ""
    assert state["final_answer"] == ""
    assert state["error"] == ""
    assert state["iteration"] == 0
    assert state["needs_graph"] is False
    assert state["graph_type"] == ""
    assert state["graph_json"] == ""
    assert state["is_in_scope"] is True


def test_sanitize_sql_response_removes_code_fences():
    raw_sql = "```sql\nSELECT * FROM orders LIMIT 5;\n```"

    assert sanitize_sql_response(raw_sql) == "SELECT * FROM orders LIMIT 5;"
