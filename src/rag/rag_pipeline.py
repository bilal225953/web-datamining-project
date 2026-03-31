"""
RAG Pipeline: Knowledge-Grounded AI Assistant
- NL → SPARQL translation using Ollama (local LLM)
- Self-repair mechanism for failed SPARQL queries
- Baseline vs RAG comparison
- CLI / interactive demo
"""

import os
import re
import json
import logging
from rdflib import Graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────
OLLAMA_MODEL = "mistral"  # or "llama3", "phi3", etc.
KB_PATH = "kg_artifacts/expanded.ttl"


def get_schema_summary(kg_path=KB_PATH):
    """Generate a schema summary of the KB for the LLM prompt."""
    g = Graph()
    g.parse(kg_path, format="turtle")

    # Get classes
    classes_q = """
    SELECT DISTINCT ?class ?label WHERE {
        ?class a owl:Class .
        OPTIONAL { ?class rdfs:label ?label }
    }
    """
    # Get properties
    props_q = """
    SELECT DISTINCT ?prop ?domain ?range WHERE {
        { ?prop a owl:ObjectProperty } UNION { ?prop a owl:DatatypeProperty }
        OPTIONAL { ?prop rdfs:domain ?domain }
        OPTIONAL { ?prop rdfs:range ?range }
    }
    """
    # Get sample instances
    instances_q = """
    SELECT ?s ?type (SAMPLE(?label) AS ?l) WHERE {
        ?s a ?type .
        OPTIONAL { ?s rdfs:label ?label }
    } GROUP BY ?s ?type LIMIT 20
    """

    schema = "=== KNOWLEDGE BASE SCHEMA ===\n\n"
    schema += "PREFIX ai: <http://example.org/ai-ontology#>\n"
    schema += "PREFIX data: <http://example.org/data/>\n\n"

    schema += "CLASSES:\n"
    for row in g.query(classes_q):
        cls_name = str(row[0]).split("#")[-1].split("/")[-1]
        label = str(row[1]) if row[1] else cls_name
        schema += f"  - ai:{cls_name}\n"

    schema += "\nPROPERTIES:\n"
    for row in g.query(props_q):
        prop = str(row[0]).split("#")[-1].split("/")[-1]
        domain = str(row[1]).split("#")[-1].split("/")[-1] if row[1] else "?"
        range_ = str(row[2]).split("#")[-1].split("/")[-1] if row[2] else "?"
        schema += f"  - ai:{prop} (domain: {domain}, range: {range_})\n"

    schema += "\nSAMPLE ENTITIES:\n"
    for row in g.query(instances_q):
        name = str(row[2]) if row[2] else str(row[0]).split("/")[-1]
        type_ = str(row[1]).split("#")[-1].split("/")[-1]
        schema += f"  - data:{name.replace(' ', '_')} (type: {type_})\n"

    return schema


def build_nl_to_sparql_prompt(question, schema_summary):
    """Build the prompt for NL→SPARQL translation."""
    prompt = f"""You are a SPARQL query generator. Given a knowledge base schema and a user question,
generate a valid SPARQL query.

{schema_summary}

IMPORTANT RULES:
- Use PREFIX ai: <http://example.org/ai-ontology#>
- Use PREFIX data: <http://example.org/data/>
- Use PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
- Use PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
- Only output the SPARQL query, no explanations
- Use FILTER(CONTAINS(LCASE(STR(?var)), "search_term")) for fuzzy matching
- Return SELECT queries with LIMIT 10

USER QUESTION: {question}

SPARQL QUERY:"""
    return prompt


def build_self_repair_prompt(question, failed_query, error_msg, schema_summary):
    """Prompt for self-repair of failed SPARQL queries."""
    prompt = f"""The following SPARQL query failed. Fix it.

{schema_summary}

ORIGINAL QUESTION: {question}

FAILED QUERY:
{failed_query}

ERROR: {error_msg}

RULES:
- Fix syntax errors
- Ensure prefixes are correct
- Use OPTIONAL for properties that may not exist
- Only output the corrected SPARQL query

CORRECTED SPARQL QUERY:"""
    return prompt


def call_ollama(prompt, model=OLLAMA_MODEL):
    """Call Ollama local LLM API."""
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 500},
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            logger.error(f"Ollama error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return None


def extract_sparql_from_response(response):
    """Extract SPARQL query from LLM response (handles markdown fences etc)."""
    if not response:
        return None
    # Try to extract from code block
    match = re.search(r"```(?:sparql)?\s*(SELECT.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try to find SELECT directly
    match = re.search(r"((?:PREFIX.*\n)*\s*SELECT.*)", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def execute_sparql(graph, query):
    """Execute a SPARQL query on the graph."""
    try:
        results = graph.query(query)
        rows = []
        for row in results:
            rows.append([str(val).split("/")[-1].split("#")[-1] if val else "" for val in row])
        return rows, None
    except Exception as e:
        return None, str(e)


def build_answer_prompt(question, sparql_results, schema_summary):
    """Build a prompt to generate a natural language answer from SPARQL results."""
    results_str = "\n".join([" | ".join(row) for row in sparql_results[:10]])
    prompt = f"""Based on the following knowledge base query results, answer the user's question
in natural language. Be concise and factual.

QUESTION: {question}

QUERY RESULTS:
{results_str}

If the results are empty, say "I don't have enough information in my knowledge base to answer that."

ANSWER:"""
    return prompt


class RAGAssistant:
    """Knowledge-grounded AI assistant using SPARQL-based RAG."""

    def __init__(self, kb_path=KB_PATH, model=OLLAMA_MODEL):
        self.model = model
        self.graph = Graph()
        self.graph.parse(kb_path, format="turtle")
        self.schema_summary = get_schema_summary(kb_path)
        self.max_repair_attempts = 2
        logger.info(f"RAG Assistant loaded with {len(self.graph)} triples")

    def ask(self, question):
        """Full RAG pipeline: NL → SPARQL → Execute → Answer."""
        result = {"question": question, "steps": []}

        # Step 1: Generate SPARQL
        prompt = build_nl_to_sparql_prompt(question, self.schema_summary)
        sparql_response = call_ollama(prompt, self.model)

        if not sparql_response:
            # Fallback: try without Ollama (hardcoded queries)
            return self._fallback_answer(question, result)

        sparql_query = extract_sparql_from_response(sparql_response)
        result["steps"].append({"step": "NL→SPARQL", "query": sparql_query})

        # Step 2: Execute SPARQL
        rows, error = execute_sparql(self.graph, sparql_query)

        # Step 3: Self-repair if failed
        repair_count = 0
        while error and repair_count < self.max_repair_attempts:
            repair_count += 1
            result["steps"].append({"step": f"Self-repair #{repair_count}", "error": error})

            repair_prompt = build_self_repair_prompt(
                question, sparql_query, error, self.schema_summary)
            repaired = call_ollama(repair_prompt, self.model)

            if repaired:
                sparql_query = extract_sparql_from_response(repaired)
                rows, error = execute_sparql(self.graph, sparql_query)
                result["steps"].append({"step": "Repaired query", "query": sparql_query})

        if error:
            result["answer"] = f"I couldn't execute the query: {error}"
            result["sparql"] = sparql_query
            return result

        result["sparql"] = sparql_query
        result["raw_results"] = rows

        # Step 4: Generate NL answer
        if rows:
            answer_prompt = build_answer_prompt(question, rows, self.schema_summary)
            answer = call_ollama(answer_prompt, self.model)
            result["answer"] = answer or f"Found {len(rows)} results: {rows[:5]}"
        else:
            result["answer"] = "No results found in the knowledge base for this question."

        return result

    def _fallback_answer(self, question, result):
        """Fallback using direct SPARQL when Ollama is unavailable."""
        q_lower = question.lower()

        # Pre-built queries for common patterns
        if "who" in q_lower and ("develop" in q_lower or "create" in q_lower):
            sparql = """
            PREFIX ai: <http://example.org/ai-ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?entity ?creator WHERE {
                ?entity ai:developedBy ?creator .
                ?creator rdfs:label ?name .
            } LIMIT 10
            """
        elif "what" in q_lower and "field" in q_lower:
            sparql = """
            PREFIX ai: <http://example.org/ai-ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?concept ?field WHERE {
                ?concept ai:partOfField ?field .
                ?field rdfs:label ?fieldName .
            } LIMIT 10
            """
        else:
            sparql = """
            PREFIX ai: <http://example.org/ai-ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o .
                FILTER(isURI(?o))
            } LIMIT 10
            """

        rows, error = execute_sparql(self.graph, sparql)
        result["sparql"] = sparql
        result["raw_results"] = rows if rows else []
        result["answer"] = f"Found {len(rows) if rows else 0} results. (Ollama unavailable, using fallback queries)"
        return result

    def baseline_answer(self, question):
        """Baseline: just ask the LLM without any KB grounding."""
        prompt = f"Answer the following question concisely:\n\n{question}\n\nAnswer:"
        answer = call_ollama(prompt, self.model)
        return answer or "Baseline unavailable (Ollama not running)"


def run_evaluation(assistant, output_path="data/rag/evaluation.json"):
    """Run the 5-question evaluation: baseline vs RAG (required by grading)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    questions = [
        "Who developed GPT-4?",
        "What field does the Transformer algorithm belong to?",
        "Which organization is Geoffrey Hinton affiliated with?",
        "What models use the Transformer architecture?",
        "What is the relationship between Deep Learning and Machine Learning?",
        "Which frameworks are used to implement neural network algorithms?",
        "What datasets is BERT trained on?",
    ]

    results = []
    for q in questions:
        logger.info(f"Evaluating: {q}")
        rag_result = assistant.ask(q)
        baseline = assistant.baseline_answer(q)

        results.append({
            "question": q,
            "rag_answer": rag_result.get("answer", ""),
            "rag_sparql": rag_result.get("sparql", ""),
            "rag_results_count": len(rag_result.get("raw_results", [])),
            "baseline_answer": baseline,
            "rag_steps": rag_result.get("steps", []),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation saved to {output_path}")
    return results


def interactive_demo():
    """Interactive CLI demo."""
    print("=" * 60)
    print("  AI Knowledge Assistant (RAG over Knowledge Graph)")
    print("=" * 60)
    print("Type your question, or 'quit' to exit.\n")

    assistant = RAGAssistant()

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        result = assistant.ask(question)
        print(f"\nAssistant: {result.get('answer', 'No answer')}")
        if result.get("sparql"):
            print(f"\n[SPARQL used]: {result['sparql'][:200]}...")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        assistant = RAGAssistant()
        run_evaluation(assistant)
    else:
        interactive_demo()
