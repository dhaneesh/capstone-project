# capstone-project

A lightweight Text-to-SQL assistant for an e-commerce dataset powered by LangGraph and OpenAI.

## Overview

This project combines a LangGraph workflow with a Chainlit chat interface so users can ask natural-language questions about e-commerce data and receive:

- a generated SQLite query
- the executed query results
- a natural-language answer
- an optional Plotly visualization

## Project structure

- app.py: Chainlit frontend for the chat experience
- text2sql_agent.py: LangGraph-based Text-to-SQL workflow
- db_init.py: SQLite database initialization script
- get_dataset.py: data download helper
- data/: sample e-commerce CSV files used to build the SQLite database

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-key"
   ```
4. Initialize the SQLite database:
   ```bash
   python db_init.py
   ```
5. Start the chat app:
   ```bash
   chainlit run app.py
   ```

## Example questions

- How many orders were delivered?
- What are the top 5 product categories by sales?
- Show me orders from São Paulo.
- What is the average review score?
- Which sellers have the most orders?

## Notes

- The workflow uses the OpenAI model configured by the OPENAI_MODEL environment variable, with a default of gpt-4o-mini.
- The SQLite database path can be customized with the DB_PATH environment variable.
- The agent expects the database file to exist before execution.
