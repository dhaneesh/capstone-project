import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text2sql_agent import (
    GuardrailsDecision,
    VisualizationSpec,
    build_plotly_figure,
    create_initial_state,
    parse_structured_output,
    sanitize_sql_response,
)


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


def test_parse_structured_output_uses_pydantic_model():
    payload = '{"is_in_scope": true, "is_greeting": false, "reason": "Relevant to orders"}'

    decision = parse_structured_output(payload, GuardrailsDecision)

    assert isinstance(decision, GuardrailsDecision)
    assert decision.is_in_scope is True
    assert decision.is_greeting is False


def test_build_plotly_figure_constructs_a_bar_chart():
    df = pd.DataFrame(
        [
            {"status": "delivered", "count": 120},
            {"status": "shipped", "count": 40},
        ]
    )
    spec = VisualizationSpec(
        chart_type="bar",
        title="Orders by status",
        x_column="status",
        y_column="count",
    )

    fig = build_plotly_figure(df, spec)

    assert fig.layout.title.text == "Orders by status"
    assert len(fig.data) == 1
