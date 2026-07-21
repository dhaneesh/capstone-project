# define the process question

def process_question(question: str) -> dict:
    """
    Process a natural language question and return the final result.
    This is a simple synchronous function for notebook usage.
    
    Args:
        question: Natural language question about the e-commerce data
        
    Returns:
        dict: Final state with answer, SQL query, and graph data if applicable
    """
    initial_state = AgentState(
        question=question,
        sql_query="",
        query_result="",
        final_answer="",
        error="",
        iteration=0,
        needs_graph=False,
        graph_type="",
        graph_json="",
        is_in_scope=True
    )
    
    try:
        # Invoke the graph
        final_state = text2sql_graph.invoke(
            initial_state,
            config={"recursion_limit": 50}
        )
        
        return final_state
        
    except Exception as e:
        return {
            "error": str(e),
            "final_answer": f"An error occurred while processing your question: {str(e)}"
        }

## Define the test function:

def test_process_question(question: str):

    result = process_question(question)
    
    print("Final Answer:")
    print(result.get("final_answer", "No answer generated."))

    if result.get("sql_query"):
    
        print("\nGenerated SQL Query:")
        print(result.get("sql_query", "No SQL query generated."))

    print(f"\nNeeds Graph: {result['needs_graph']}")
    if result['needs_graph']:
        print(f"Graph Type: {result['graph_type']}")
    
    if result.get("graph_json"):
        import plotly.graph_objects as go
        import plotly.io as pio
    
        # Load the figure from JSON
        fig = pio.from_json(result['graph_json'])
        
        # Display in notebook
        fig.show()
    else:
        print("No graph was generated for this question.")