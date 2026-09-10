"""
13_schema_linking.py
What this does: schema linking module, replicating the mechanism from
"Rethinking Schema Linking: A Context-Aware Bidirectional Retrieval Approach
for Text-to-SQL" -- instead of always handing the LLM the full database
schema, first retrieve just the tables/columns a question actually needs,
so the SQL-generation prompt only contains relevant schema information.

Two pieces:
1. extract_gold_schema() -- given a gold (known-correct) SQL query, works
   out exactly which tables/columns it uses. This gives us a "Perfect"
   oracle schema (the theoretical best case: what if retrieval were
   flawless?) and a ground truth to measure our real retrieval against.
2. retrieve_schema() -- the actual bidirectional retrieval mechanism used
   at question-answering time (no access to the gold SQL): question
   augmentation (keyword/keyphrase extraction + subquestion decomposition),
   then Table-First retrieval (which whole tables are relevant) and
   Column-First retrieval (which specific columns are relevant, across all
   tables), unioned together. Table-First and Column-First catch different
   things -- Table-First is stronger on broad entity references ("list all
   books..."), Column-First is stronger on attribute-driven questions
   ("highest average rating") and can recover tables whose relevance is
   only clear from a specific column. Each step is one LLM call using the
   paper's own prompt structure (Appendix G), for four calls total per
   question.

Uses gpt-4o-mini via open_ai.chat_gpt.ask_chatGPT for every retrieval call,
plus one more call for the final SQL generation itself (see
13b_generate_schema_linked.py) -- 5 LLM calls per question.
"""
import json
import re
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope
from open_ai.chat_gpt import ask_chatGPT


# ---------------------------------------------------------------------------
# Part 1: gold/oracle schema extraction
# ---------------------------------------------------------------------------

def extract_gold_schema(gold_sql, schema):
    """
    gold_sql: the gold SQL string (SQLite dialect).
    schema: {table_lower: [col_lower, ...]} as returned by
            eval.process_sql.get_schema(db_path).
    Returns: {table_lower: set(col_lower)} -- the minimal schema the query
             needs. A table can map to an empty set if it's needed
             structurally (e.g. for COUNT(*) or a join) but no specific
             column is referenced.

    Uses sqlglot's optimizer.scope (build_scope) to resolve each column
    reference to its source table per nesting level -- handles aliases,
    JOINs, and subqueries correctly. Every candidate (table, column) is
    cross-checked against the real schema dict before being kept.
    """
    gold = {}
    try:
        tree = sqlglot.parse_one(gold_sql, read="sqlite")
        root = build_scope(tree)
    except Exception:
        return gold
    if root is None:
        return gold

    for scope in root.traverse():
        table_sources = {
            alias.lower(): source.name.lower()
            for alias, source in scope.sources.items()
            if isinstance(source, exp.Table) and source.name.lower() in schema
        }
        single_table = list(table_sources.values())[0] if len(table_sources) == 1 else None

        for real_table in table_sources.values():
            gold.setdefault(real_table, set())

        for col in scope.columns:
            col_name = col.name.lower()
            table_token = (col.table or "").lower()
            real_table = table_sources.get(table_token) or single_table
            if real_table and col_name in schema.get(real_table, []):
                gold[real_table].add(col_name)

    return gold


# ---------------------------------------------------------------------------
# Part 2: the actual bidirectional retrieval mechanism (used at question time,
# no access to gold SQL)
# ---------------------------------------------------------------------------

def parse_json_response(text):
    """Best-effort JSON parsing of an LLM text response: strips markdown
    code fences if present, falls back to extracting the first {...} block
    if the model added stray commentary. Returns None if nothing parses."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def extract_keywords(question, hint=""):
    """Keyword/Keyphrase Extraction (question augmentation, step 1 of 2)."""
    prompt = f'''Objective: Analyze the given question and hint to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry and the guidance provided. The goal is to recognize and isolate significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

Instructions:
1. Read the Question Carefully: Identify the main focus, named entities (e.g., organizations, locations), technical terms, and key concepts.
2. Analyze the Hint: Extract keywords or phrases that provide clarity or direction toward answering the question.
3. List Keyphrases and Entities: Combine findings from both the question and hint into a single Python list. Include:
- Keywords: Essential single words.
- Keyphrases: Multi-word terms or named entities.
Ensure all terms maintain the original phrasing or terminology used in the input.

Task:
Given the following question and hint, identify and list all relevant keywords, keyphrases, and named entities.
Question: {question}
Hint: {hint}

Please provide your findings as a json file, capturing the essence of both the question and hint through the identified terms and phrases.
{{
"keywords": list of keywords, keyphrases and entities.
}}
Do not include ```json in your response. Only output a json object as your response.'''
    result = parse_json_response(ask_chatGPT(prompt))
    if result and isinstance(result.get("keywords"), list):
        return result["keywords"]
    return []


def decompose_question(question):
    """Question Decomposition into subquestions (question augmentation, step 2 of 2)."""
    prompt = f'''You are tasked with decomposing a natural language question into smaller subquestions to help generate an SQL query from a database.

Your goal is to break the question into simple, specific subquestions that together fully address the original question. Each subquestion should focus on a single aspect or step needed to answer the original question, such as identifying data, filtering, or calculating something.

Here are some examples:
Example 1:
Original Question: "What is the total revenue from orders placed in 2023?"
Subquestions:
1. "Which orders were placed in 2023?"
2. "What is the revenue for each of those orders?"
3. "What is the total of that revenue?"

Example 2:
Original Question: "Which employees work in departments located in New York?"
Subquestions:
1. "Which departments are located in New York?"
2. "Which employees work in those departments?"

Now, decompose the following question into subquestions. Provide the subquestions as a numbered list.
Original Question: {question}

Provide the subquestions in a json object without any explanation. Please respond with a JSON object structured as follows:
{{
"Subquestions": list of subquestions.
}}
Do not include ```json in your response. Only output a json object as your response.'''
    result = parse_json_response(ask_chatGPT(prompt))
    if result and isinstance(result.get("Subquestions"), list):
        return result["Subquestions"]
    return []


def _format_question_info(question, subquestions, keywords):
    return (
        f"Original Question: {question}\n"
        f"Sub-questions: {subquestions}\n"
        f"keywords and key-phrases: {keywords}"
    )


def _clean_schema_response(result, schema):
    """Validates an LLM JSON response against the real schema (case-
    insensitive), drops the 'chain of thought reasoning' key and any
    hallucinated tables/columns that aren't actually in the database."""
    if not result or not isinstance(result, dict):
        return {}
    out = {}
    for k, v in result.items():
        if k.lower() in ("chain of thought reasoning", "reasoning"):
            continue
        t = k.lower()
        if t not in schema or not isinstance(v, list):
            continue
        cols = [c.lower() for c in v if isinstance(c, str) and c.lower() in schema[t]]
        out[t] = cols
    return out


def retrieve_tables(question, subquestions, keywords, schema, hint=""):
    """Table-First retrieval: selects whole tables, keeping all their columns."""
    augmented_question = _format_question_info(question, subquestions, keywords)
    prompt = f'''You are tasked with identifying the relevant tables and all their columns from a database schema needed to write an SQL query from a natural language question. The schema is provided as a dictionary where keys are table names and values are lists of column names. You will receive:
- The original question
- A list of subquestions breaking down the original question into simpler parts
- A list of keywords and keyphrases extracted from the question
- The schema containing tables and their full column lists
- An optional hint (if provided)

Your goal is to select the tables required to construct an SQL query that answers the question and include all columns from those tables. To ensure high recall, include any table that might be relevant based on the original question, subquestions, keywords, or hint, even if its relevance is uncertain.

Output the result as a JSON-formatted dictionary where keys are the selected table names and values are their full column lists, like {{"table1": ["col1"], "table2": ["col2"]}}.

Now, identify the relevant columns for the following:
Schema: {schema}
Question Information: {augmented_question}
Hint: {hint}

Please respond with a JSON object structured as follows:
{{
"chain of thought reasoning": "Your reasoning for selecting the columns, be concise and clear.",
"table name1": ["column1", "column2"],
"table name2": ["column1", "column2"]
}}
Make sure your response includes the table names as keys, each associated with a list of column names that are necessary for writing a SQL query to answer the original question. Be cautious about the foreign keys.
Do not include ```json in your response. Only output a json as your response.'''
    result = parse_json_response(ask_chatGPT(prompt))
    return _clean_schema_response(result, schema)


def retrieve_columns(question, subquestions, keywords, schema, hint=""):
    """Column-First retrieval: selects specific columns per table."""
    augmented_question = _format_question_info(question, subquestions, keywords)
    prompt = f'''You are tasked with identifying the relevant columns from a database schema needed to write an SQL query form a natural language question. The schema is provided as a dictionary where keys are table names and values are lists of column names (e.g., {{"Employees": ["ID", "Name"]}}). You will receive:
- The original question
- A list of subquestions breaking down the original question into simpler parts
- A list of keywords and keyphrases extracted from the question
- The schema containing tables and their full column lists
- An optional hint (if provided)

Your goal is to select the columns from each table required to construct an SQL query that answers the question. To ensure high recall, include any column that might be relevant based on the original question, subquestions, or keywords, such as columns for filtering, joining, or computing results.

Output the result as a JSON-formatted dictionary where keys are the table names from the filtered schema and values are the selected column names, like {{"table1": ["col1"], "table2": ["col2"]}}.

Now, identify the relevant columns for the following:
Schema: {schema}
Question Information: {augmented_question}
Hint: {hint}

Please respond with a JSON object structured as follows:
{{
"chain of thought reasoning": "Your reasoning for selecting the columns, be concise and clear.",
"table name1": ["column1", "column2"],
"table name2": ["column1", "column2"]
}}
Make sure your response includes the table names as keys, each associated with a list of column names that are necessary for writing a SQL query to answer the question. Be cautious about the foreign keys.
Do not include ```json in your response. Only output a json as your response.'''
    result = parse_json_response(ask_chatGPT(prompt))
    return _clean_schema_response(result, schema)


def retrieve_schema(question, schema, hint=""):
    """
    Full bidirectional schema linking for one question (4 LLM calls):
    question augmentation (keywords + subquestions), then Table-First +
    Column-First retrieval, unioned together. Returns {table: [col, ...]},
    same format as eval.process_sql.get_schema() -- ready to drop straight
    into the SQL-generation prompt.
    """
    keywords = extract_keywords(question, hint)
    subquestions = decompose_question(question)

    table_first = retrieve_tables(question, subquestions, keywords, schema, hint)
    column_first = retrieve_columns(question, subquestions, keywords, schema, hint)

    selected = {}
    for t in table_first:
        selected[t] = set(schema[t])  # Table-First keeps ALL columns of a selected table
    for t, cols in column_first.items():
        selected.setdefault(t, set()).update(cols)

    return {t: [c for c in schema[t] if c in cols] for t, cols in selected.items()}
