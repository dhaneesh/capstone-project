"""Text-to-SQL workflow for e-commerce analytics.

This module provides a small LangGraph workflow that accepts a natural language
question, validates scope, generates SQLite SQL, executes it against the local
SQLite database, explains the result, and optionally creates a Plotly chart.

The implementation uses a compatibility layer around the latest OpenAI Responses
API when available and falls back to the classic chat completions API when
needed.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Literal, TypeVar, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
import plotly.express as px

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client: OpenAI | None = None
async_client: AsyncOpenAI | None = None
DB_PATH = os.getenv("DB_PATH", "ecommerce.db")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TModel = TypeVar("TModel", bound=BaseModel)


class AgentState(TypedDict, total=False):
    """State of the Text2SQL agent workflow."""

    question: str
    sql_query: str
    query_result: str
    final_answer: str
    error: str
    iteration: int
    needs_graph: bool
    graph_type: str
    graph_json: str
    is_in_scope: bool


class GuardrailsDecision(BaseModel):
    """Structured output for the guardrails agent."""

    is_in_scope: bool
    is_greeting: bool = False
    reason: str


class GraphDecision(BaseModel):
    """Structured output for the graph decision agent."""

    needs_graph: bool = False
    graph_type: Literal["bar", "line", "pie", "scatter", "none"] = "none"
    reason: str = ""


class VisualizationSpec(BaseModel):
    """Structured output describing the Plotly figure to construct."""

    chart_type: Literal["bar", "line", "pie", "scatter"] = "bar"
    title: str = ""
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None


SCHEMA_INFO = """
Database Schema for E-commerce System:

1. customers
   - customer_id (TEXT): Unique customer identifier
   - customer_unique_id (TEXT): Unique customer identifier across datasets
   - customer_zip_code_prefix (INTEGER): Customer zip code
   - customer_city (TEXT): Customer city
   - customer_state (TEXT): Customer state

2. orders
   - order_id (TEXT): Unique order identifier
   - customer_id (TEXT): Foreign key to customers
   - order_status (TEXT): Order status (delivered, shipped, etc.)
   - order_purchase_timestamp (TEXT): When the order was placed
   - order_approved_at (TEXT): When payment was approved
   - order_delivered_carrier_date (TEXT): When order was handed to carrier
   - order_delivered_customer_date (TEXT): When customer received the order
   - order_estimated_delivery_date (TEXT): Estimated delivery date

3. order_items
   - order_id (TEXT): Foreign key to orders
   - order_item_id (INTEGER): Item sequence number within order
   - product_id (TEXT): Foreign key to products
   - seller_id (TEXT): Foreign key to sellers
   - shipping_limit_date (TEXT): Shipping deadline
   - price (REAL): Item price
   - freight_value (REAL): Shipping cost

4. order_payments
   - order_id (TEXT): Foreign key to orders
   - payment_sequential (INTEGER): Payment sequence number
   - payment_type (TEXT): Payment method (credit_card, boleto, etc.)
   - payment_installments (INTEGER): Number of installments
   - payment_value (REAL): Payment amount

5. order_reviews
   - review_id (TEXT): Unique review identifier
   - order_id (TEXT): Foreign key to orders
   - review_score (INTEGER): Review score (1-5)
   - review_comment_title (TEXT): Review title
   - review_comment_message (TEXT): Review message
   - review_creation_date (TEXT): When review was created
   - review_answer_timestamp (TEXT): When review was answered

6. products
   - product_id (TEXT): Unique product identifier
   - product_category_name (TEXT): Product category (in Portuguese)
   - product_name_lenght (REAL): Product name length
   - product_description_lenght (REAL): Product description length
   - product_photos_qty (REAL): Number of product photos
   - product_weight_g (REAL): Product weight in grams
   - product_length_cm (REAL): Product length in cm
   - product_height_cm (REAL): Product height in cm
   - product_width_cm (REAL): Product width in cm

7. sellers
   - seller_id (TEXT): Unique seller identifier
   - seller_zip_code_prefix (INTEGER): Seller zip code
   - seller_city (TEXT): Seller city
   - seller_state (TEXT): Seller state

8. geolocation
   - geolocation_zip_code_prefix (INTEGER): Zip code prefix
   - geolocation_lat (REAL): Latitude
   - geolocation_lng (REAL): Longitude
   - geolocation_city (TEXT): City name
   - geolocation_state (TEXT): State code

9. product_category_name_translation
   - product_category_name (TEXT): Category name in Portuguese
   - product_category_name_english (TEXT): Category name in English
"""

AGENT_CONFIG = {
    "guardrails_agent": {
        "role": "Security and Scope Manager",
        "system_prompt": "You are a strict guardrails system that filters questions to ensure they are relevant to e-commerce data analysis or identifies greetings.",
    },
    "sql_agent": {
        "role": "SQL Expert",
        "system_prompt": "You are a senior SQL developer specializing in e-commerce databases. Generate only valid SQLite queries without any formatting or explanation.",
    },
    "analysis_agent": {
        "role": "Data Analyst",
        "system_prompt": "You are a helpful data analyst that explains database query results in natural language with clear insights.",
    },
    "viz_agent": {
        "role": "Visualization Specialist",
        "system_prompt": "You are a data visualization expert. Return a structured Plotly chart specification without markdown formatting or explanation.",
    },
    "error_agent": {
        "role": "Error Recovery Specialist",
        "system_prompt": "You diagnose and fix SQL errors with expert knowledge of database schemas and query optimization.",
    },
}


def get_openai_client() -> OpenAI:
    """Create the OpenAI client lazily so the module can be imported safely."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Set it before running the agent.")
        client = OpenAI(api_key=api_key)
    return client


def get_async_openai_client() -> AsyncOpenAI:
    """Create the asynchronous OpenAI client lazily so async workflows can use it."""
    global async_client
    if async_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Set it before running the agent.")
        async_client = AsyncOpenAI(api_key=api_key)
    return async_client


GUARDRAILS_PROMPT_TEMPLATE = """You are a guardrails system for an e-commerce database chatbot. Your job is to determine if a user's question is related to e-commerce data, if it's a greeting, or if it's out of scope.

The chatbot has access to an e-commerce database with information about:
- Customers and their locations
- Orders and order status (data from 2016-2018)
- Products and categories
- Sellers
- Payments
- Reviews
- Shipping and delivery information

Examples of GREETING messages:
- "Hi", "Hello", "Hey"
- "Good morning", "Good afternoon"
- "How are you?"
- Any casual greeting or introduction

Examples of IN-SCOPE questions:
- "How many orders were placed last month?"
- "What are the top selling products?"
- "Show me customer distribution by state"
- "What is the average order value?"
- "Which sellers have the highest ratings?"

Examples of OUT-OF-SCOPE questions:
- Personal questions (e.g., "What is my wife's name?", "Where do I live?")
- Political questions (e.g., "Who should I vote for?", "What do you think about the president?")
- General knowledge (e.g., "What is the capital of France?", "How does photosynthesis work?")
- Unrelated topics (e.g., "Tell me a joke", "What's the weather like?")

User Question: {question}

Analyze the question and respond in JSON format with the required fields:
{{
    "is_in_scope": true/false,
    "is_greeting": true/false,
    "reason": "brief explanation of why it is or isn't in scope or if it's a greeting"
}}

If the question is a greeting, mark is_greeting as true and is_in_scope as false.
If the question is ambiguous but could potentially relate to the e-commerce data, mark it as in_scope.
"""

SQL_PROMPT_TEMPLATE = """You are a SQL expert. Convert the following natural language question into a valid SQLite query.

{schema_info}

Question: {question}

Important Guidelines:
1. Use only the tables and columns mentioned in the schema.
2. Use proper JOIN clauses when querying multiple tables.
3. Return ONLY the SQL query without any explanation or markdown formatting.
4. If the question contains multiple sub-questions, generate separate SQL queries separated by semicolons.
5. Use aggregate functions (COUNT, SUM, AVG, etc.) appropriately.
6. Add LIMIT clauses for queries that might return many rows (default LIMIT 10 unless user specifies).
7. Use proper WHERE clauses to filter data.
8. For date comparisons, the dates are stored as TEXT in ISO format.
9. Each SQL statement should be on its own line for clarity when multiple queries are needed.

Generate the SQL query:"""

ERROR_PROMPT_TEMPLATE = """The following SQL query failed with an error. Please fix it.

{schema_info}

Original Question: {question}

Failed SQL Query: {sql_query}

Error: {error}

Generate a corrected SQL query that will work. Return ONLY the SQL query without any explanation or markdown formatting:"""

ANALYSIS_PROMPT_TEMPLATE = """You are a helpful assistant that explains database query results in natural language.

Original Question: {question}

SQL Query Used: {sql_query}

Query Results:
{query_result}

Please provide a clear, concise answer to the original question based on the query results.
Format the answer in a user-friendly way. If the results contain numbers, present them clearly.
If there are multiple queries/results (for multi-part questions), address each part of the question separately.
Use bullet points or numbered lists for multiple answers.

Answer:"""

GRAPH_DECISION_PROMPT_TEMPLATE = """Analyze the following question and query results to determine if a graph visualization would be helpful.

Question: {question}

Query Results Sample:
{query_result}

Determine:
1. Would a graph be helpful for this data? (YES/NO)
2. If yes, what type of graph? (bar, line, pie, scatter)

Consider:
- Trends over time → line chart
- Comparisons between categories → bar chart
- Proportions/percentages → pie chart
- Correlations → scatter plot
- Simple counts or single values → NO graph needed

Respond in JSON format:
{{"needs_graph": true/false, "graph_type": "bar/line/pie/scatter/none", "reason": "brief explanation"}}
"""

VIZ_SPEC_PROMPT_TEMPLATE = """Return a JSON object describing the Plotly chart to generate for the following data.

Question: {question}
Graph Type: {graph_type}
Columns: {columns}
Sample Data (first 5 rows): {sample_data}
Total Rows: {row_count}

Return ONLY a JSON object with these fields:
{{
  "chart_type": "bar|line|pie|scatter",
  "title": "short chart title",
  "x_column": "name of the x-axis column",
  "y_column": "name of the y-axis column",
  "color_column": "optional column for color grouping"
}}
"""


def get_openai_client() -> OpenAI:
    """Create the OpenAI client lazily so the module can be imported safely."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Set it before running the agent.")
        client = OpenAI(api_key=api_key)
    return client


def get_async_openai_client() -> AsyncOpenAI:
    """Create the asynchronous OpenAI client lazily so async workflows can use it."""
    global async_client
    if async_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Set it before running the agent.")
        async_client = AsyncOpenAI(api_key=api_key)
    return async_client


def parse_structured_output(payload: str | dict[str, Any], model_cls: type[TModel]) -> TModel:
    """Parse JSON payload into its corresponding Pydantic model."""
    if isinstance(payload, str):
        parsed_payload = json.loads(payload)
    else:
        parsed_payload = payload

    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(parsed_payload)
    return model_cls.parse_obj(parsed_payload)


def call_openai_model(
    *,
    system_prompt: str,
    user_prompt: str,
    response_format: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    response_model: type[TModel] | None = None,
) -> str | TModel:
    """Call the OpenAI model using the latest Responses API when available."""
    openai_client = get_openai_client()

    try:
        kwargs: dict[str, Any] = {
            "model": model or DEFAULT_MODEL,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if response_format == "json_object":
            kwargs["text"] = {"format": {"type": "json_object"}}

        response = openai_client.responses.create(**kwargs)
        response_text = getattr(response, "output_text", "").strip()
    except Exception:
        kwargs = {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        response = openai_client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content.strip()

    if response_model is None:
        return response_text
    return parse_structured_output(response_text, response_model)


async def async_call_openai_model(
    *,
    system_prompt: str,
    user_prompt: str,
    response_format: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    response_model: type[TModel] | None = None,
) -> str | TModel:
    """Asynchronously call the OpenAI model and optionally parse it into a Pydantic model."""
    openai_client = get_async_openai_client()

    try:
        kwargs: dict[str, Any] = {
            "model": model or DEFAULT_MODEL,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if response_format == "json_object":
            kwargs["text"] = {"format": {"type": "json_object"}}

        response = await openai_client.responses.create(**kwargs)
        response_text = getattr(response, "output_text", "").strip()
    except Exception:
        kwargs = {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        response = await openai_client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content.strip()
    logger.info("OpenAI full response: %s", response)
    logger.info("OpenAI model response: %s", response_text)
    logger.info("OpenAI model response model: %s", response_model)
    if response_model is None:
        return response_text
    return parse_structured_output(response_text, response_model)


def sanitize_sql_response(raw_sql: str) -> str:
    """Remove markdown fences and surrounding whitespace from generated SQL."""
    return raw_sql.replace("```sql", "").replace("```", "").strip()


def create_initial_state(question: str) -> AgentState:
    """Create a fresh workflow state for a new question."""
    return {
        "question": question,
        "sql_query": "",
        "query_result": "",
        "final_answer": "",
        "error": "",
        "iteration": 0,
        "needs_graph": False,
        "graph_type": "",
        "graph_json": "",
        "is_in_scope": True,
    }


async def guardrails_agent(state: AgentState) -> AgentState:
    """Check if the question is within scope and handle greetings."""
    question = state["question"]

    prompt = GUARDRAILS_PROMPT_TEMPLATE.format(question=question)

    result = await async_call_openai_model(
        system_prompt=AGENT_CONFIG["guardrails_agent"]["system_prompt"],
        user_prompt=prompt,
        response_format="json_object",
        response_model=GuardrailsDecision,
    )
    logger.info("This is result from Guard Rails Agent: %s", result)
    assert isinstance(result, GuardrailsDecision)
    state["is_in_scope"] = result.is_in_scope
    is_greeting = result.is_greeting

    if is_greeting:
        state["final_answer"] = (
            "Hi! I am your e-commerce assistant. I can answer queries about orders, "
            "customers, products, sellers, payments, and reviews between 2016 and 2018."
        )
        return state

    if not state["is_in_scope"]:
        state["final_answer"] = (
            "I apologize, but your question appears to be out of scope. I can only answer "
            "questions about the e-commerce data, including customer information, orders, "
            "products, sellers, payments, reviews, and shipping data."
        )

    logger.info("Guardrails agent processed question: %s", question)
    return state


async def sql_agent(state: AgentState) -> AgentState:
    """Generate a SQLite query from a natural language question."""
    question = state["question"]
    iteration = state.get("iteration", 0)

    prompt = SQL_PROMPT_TEMPLATE.format(schema_info=SCHEMA_INFO, question=question)

    raw_sql = await async_call_openai_model(
        system_prompt=AGENT_CONFIG["sql_agent"]["system_prompt"],
        user_prompt=prompt,
    )
    if not isinstance(raw_sql, str):
        raw_sql = str(raw_sql)
    sql_query = sanitize_sql_response(raw_sql)
    state["sql_query"] = sql_query
    state["iteration"] = iteration + 1
    return state


async def execute_sql(state: AgentState) -> AgentState:
    """Execute the generated SQL statement(s) against the local SQLite database."""
    sql_query = state["sql_query"]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            sql_statements = [stmt.strip() for stmt in sql_query.split(";") if stmt.strip()]
            all_results: list[dict[str, Any]] | list[dict[str, Any]] = []

            for index, statement in enumerate(sql_statements):
                cursor.execute(statement)
                results = cursor.fetchall()

                if results:
                    column_names = [description[0] for description in cursor.description]
                    formatted_results = []
                    for row in results[:100]:
                        formatted_results.append(dict(zip(column_names, row)))

                    if len(sql_statements) > 1:
                        all_results.append({
                            f"query_{index + 1}": formatted_results,
                            f"query_{index + 1}_sql": statement,
                        })
                    else:
                        all_results = formatted_results

        if not all_results:
            state["query_result"] = "No results found."
        else:
            state["query_result"] = json.dumps(all_results, indent=2)

        state["error"] = ""
    except Exception as exc:  # pragma: no cover - defensive path
        state["error"] = f"SQL Execution Error: {exc}"
        state["query_result"] = ""

    return state


async def error_agent(state: AgentState) -> AgentState:
    """Retry the SQL generation workflow after an execution error."""
    error = state["error"]
    sql_query = state["sql_query"]
    question = state["question"]
    iteration = state.get("iteration", 0)

    if iteration > 3:
        state["final_answer"] = (
            f"I apologize, but I am having trouble generating a correct SQL query for your question. "
            f"Error: {error}"
        )
        return state

    prompt = ERROR_PROMPT_TEMPLATE.format(
        schema_info=SCHEMA_INFO,
        question=question,
        sql_query=sql_query,
        error=error,
    )

    corrected_query = sanitize_sql_response(
        await async_call_openai_model(
            system_prompt=AGENT_CONFIG["error_agent"]["system_prompt"],
            user_prompt=prompt,
        )
    )
    state["sql_query"] = corrected_query
    state["error"] = ""
    state["iteration"] = iteration + 1
    return state


async def analysis_agent(state: AgentState) -> AgentState:
    """Generate a human-friendly explanation from the query results."""
    question = state["question"]
    sql_query = state["sql_query"]
    query_result = state["query_result"]

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        question=question,
        sql_query=sql_query,
        query_result=query_result,
    )

    final_answer = await async_call_openai_model(
        system_prompt=AGENT_CONFIG["analysis_agent"]["system_prompt"],
        user_prompt=prompt,
    )
    if not isinstance(final_answer, str):
        final_answer = str(final_answer)
    state["final_answer"] = final_answer.strip()
    state["final_answer"] = final_answer
    return state


async def decide_graph_need(state: AgentState) -> AgentState:
    """Decide whether a Plotly chart would add value to the answer."""
    question = state["question"]
    query_result = state["query_result"]

    if not query_result or query_result == "No results found." or state.get("error"):
        state["needs_graph"] = False
        state["graph_type"] = ""
        return state

    prompt = GRAPH_DECISION_PROMPT_TEMPLATE.format(
        question=question,
        query_result=query_result[:500],
    )

    decision = await async_call_openai_model(
        system_prompt="You are a data visualization expert. Analyze queries and determine if visualization would add value.",
        user_prompt=prompt,
        response_format="json_object",
        response_model=GraphDecision,
    )
    assert isinstance(decision, GraphDecision)
    state["needs_graph"] = decision.needs_graph
    state["graph_type"] = decision.graph_type
    return state


def build_plotly_figure(df: pd.DataFrame, spec: VisualizationSpec):
    """Construct a Plotly figure directly from a structured visualization specification."""
    if df.empty:
        raise ValueError("Cannot build a chart from an empty dataframe")

    chart_type = spec.chart_type
    title = spec.title or f"{chart_type.title()} Chart"

    x_column = spec.x_column or df.columns[0]
    y_column = spec.y_column or df.columns[1] if len(df.columns) > 1 else None

    if chart_type == "pie":
        if x_column is None:
            raise ValueError("Pie charts require an x-column")
        if y_column is None:
            values = df[x_column].value_counts().reset_index()
            values.columns = [x_column, "count"]
            x_column = x_column
            y_column = "count"
        else:
            values = df[[x_column, y_column]].copy()
            values = values.rename(columns={x_column: "label", y_column: "value"})
            return px.pie(values, names="label", values="value", title=title)

    if chart_type == "bar":
        fig = px.bar(df, x=x_column, y=y_column, color=spec.color_column, title=title)
    elif chart_type == "line":
        fig = px.line(df, x=x_column, y=y_column, color=spec.color_column, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, color=spec.color_column, title=title)
    else:
        fig = px.bar(df, x=x_column, y=y_column, color=spec.color_column, title=title)

    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


async def viz_agent(state: AgentState) -> AgentState:
    """Generate a Plotly figure from the query results using a structured spec."""
    query_result = state["query_result"]
    graph_type = state["graph_type"]
    question = state["question"]

    try:
        results = json.loads(query_result)
        if not results or len(results) == 0:
            state["graph_json"] = ""
            return state

        df = pd.DataFrame(results)
        columns = df.columns.tolist()
        sample_data = df.head(5).to_dict("records")

        prompt = VIZ_SPEC_PROMPT_TEMPLATE.format(
            question=question,
            graph_type=graph_type,
            columns=columns,
            sample_data=json.dumps(sample_data, indent=2),
            row_count=len(df),
        )

        spec = await async_call_openai_model(
            system_prompt=AGENT_CONFIG["viz_agent"]["system_prompt"],
            user_prompt=prompt,
            response_format="json_object",
            response_model=VisualizationSpec,
        )
        if not isinstance(spec, VisualizationSpec):
            spec = parse_structured_output(spec, VisualizationSpec)

        fig = build_plotly_figure(df, spec)
        state["graph_json"] = fig.to_json()
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Graph generation failed: %s", exc)
        state["graph_json"] = ""

    return state


def should_retry(state: AgentState) -> str:
    """Decide whether to retry after an execution error."""
    if state.get("error"):
        iteration = state.get("iteration", 0)
        if iteration <= 3:
            return "retry"
        return "end"
    return "success"


def should_generate_graph(state: AgentState) -> str:
    """Decide whether to generate a graph for the answer."""
    if state.get("needs_graph", False):
        return "viz_agent"
    return "skip_graph"


def check_scope(state: AgentState) -> str:
    """Check whether the question is within scope before continuing."""
    if state.get("is_in_scope", True):
        return "in_scope"
    return "out_of_scope"


def create_text2sql_graph():
    """Create the LangGraph workflow for the Text2SQL experience."""
    workflow = StateGraph(AgentState)
    workflow.add_node("guardrails_agent", guardrails_agent)
    workflow.add_node("sql_agent", sql_agent)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("analysis_agent", analysis_agent)
    workflow.add_node("error_agent", error_agent)
    workflow.add_node("decide_graph_need", decide_graph_need)
    workflow.add_node("viz_agent", viz_agent)

    workflow.set_entry_point("guardrails_agent")

    workflow.add_conditional_edges(
        "guardrails_agent",
        check_scope,
        {"in_scope": "sql_agent", "out_of_scope": END},
    )
    workflow.add_edge("sql_agent", "execute_sql")
    workflow.add_conditional_edges(
        "execute_sql",
        should_retry,
        {"success": "analysis_agent", "retry": "error_agent", "end": "analysis_agent"},
    )
    workflow.add_edge("error_agent", "execute_sql")
    workflow.add_edge("analysis_agent", "decide_graph_need")
    workflow.add_conditional_edges(
        "decide_graph_need",
        should_generate_graph,
        {"viz_agent": "viz_agent", "skip_graph": END},
    )
    workflow.add_edge("viz_agent", END)

    return workflow.compile()


text2sql_graph = create_text2sql_graph()
logger.info("Compiled workflow graph")


def generate_graph_visualization(output_path: str = "text2sql_workflow.png") -> str | None:
    """Render a PNG diagram of the workflow graph to disk."""
    try:
        graph_image = text2sql_graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as handle:
            handle.write(graph_image)
        logger.info("Workflow diagram saved to %s", output_path)
        return output_path
    except Exception as exc:  # pragma: no cover - dependency optional
        logger.warning("Could not generate workflow diagram: %s", exc)
        return None


async def process_question_stream(question: str):
    """Process a question while streaming workflow events to the UI."""
    initial_state = create_initial_state(question)
    current_state: dict[str, Any] = dict(initial_state)

    try:
        async for event in text2sql_graph.astream_events(
            initial_state,
            config={"recursion_limit": 50},
            version="v2",
        ):
            event_type = event.get("event")
            node_name = event.get("name", "")
            node_alias = {
                "sql_agent": "generate_sql",
                "execute_sql": "execute_sql",
                "analysis_agent": "generate_answer",
                "error_agent": "handle_error",
                "decide_graph_need": "decide_graph_need",
                "viz_agent": "generate_graph",
                "guardrails_agent": "guardrails_agent",
            }.get(node_name, node_name)

            if event_type == "on_chain_start":
                if node_name in {
                    "guardrails_agent",
                    "sql_agent",
                    "execute_sql",
                    "analysis_agent",
                    "error_agent",
                    "decide_graph_need",
                    "viz_agent",
                }:
                    yield {"type": "node_start", "node": node_alias, "input": current_state}

            elif event_type == "on_chain_end":
                if node_name in {
                    "guardrails_agent",
                    "sql_agent",
                    "execute_sql",
                    "analysis_agent",
                    "error_agent",
                    "decide_graph_need",
                    "viz_agent",
                }:
                    output = event.get("data", {}).get("output", {})
                    if output:
                        current_state.update(output)
                        yield {
                            "type": "node_end",
                            "node": node_alias,
                            "output": output,
                            "state": current_state.copy(),
                        }

        yield {"type": "final", "result": current_state}
    except Exception as exc:  # pragma: no cover - defensive path
        yield {"type": "error", "error": str(exc)}


if __name__ == "__main__":
    print("=" * 80)
    print("Text2SQL Agent - Use 'chainlit run app.py' to start the web interface")
    print("=" * 80)
    print("\nThis module is meant to be imported and used via the Chainlit app.")
    print("Run: chainlit run app.py")
