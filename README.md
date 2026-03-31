# Web Datamining & Semantics — AI Knowledge Assistant

> **ESILV M1 — Web Datamining and Semantics Project**
> Building a Knowledge-Supported AI Assistant: from raw web data to a RAG system grounded in a Knowledge Graph.

## Overview

This project implements an end-to-end pipeline that:
1. **Crawls** Wikipedia articles about Artificial Intelligence
2. **Extracts** entities (NER) and relations from text
3. **Builds** an RDF Knowledge Graph with a domain ontology
4. **Reasons** over the KB using SWRL rules (OWLReady2)
5. **Trains** Knowledge Graph Embeddings (TransE, ComplEx)
6. **Serves** a RAG assistant that translates natural language to SPARQL

## Project Structure

```
project-root/
├── src/
│   ├── crawl/         # Web crawler (trafilatura + requests)
│   ├── ie/            # NER & Information Extraction (spaCy)
│   ├── kg/            # RDF/OWL Knowledge Graph builder (rdflib)
│   ├── reason/        # SWRL reasoning (OWLReady2)
│   ├── kge/           # Knowledge Graph Embeddings (PyKEEN)
│   └── rag/           # RAG pipeline (NL→SPARQL + Ollama)
├── data/
│   ├── raw/           # Crawled articles (JSON)
│   ├── processed/     # Cleaned text, entities, triples
│   ├── kge/           # train/valid/test splits
│   └── reasoning/     # SWRL outputs
├── kg_artifacts/
│   ├── ontology.ttl   # Domain ontology (OWL)
│   ├── alignment.ttl  # Alignment to schema.org/DBpedia
│   ├── knowledge_graph.ttl
│   └── expanded.ttl   # Expanded KB after SPARQL inference
├── reports/
│   └── final_report.pdf
├── run_pipeline.py    # Main runner (full pipeline)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- Java (for OWLReady2 reasoner — optional)
- [Ollama](https://ollama.ai/) (for the RAG LLM — used with Mistral model)

### Setup

```bash
# Clone the repository
git clone https://github.com/bilal225953/web-datamining-project.git
cd web-datamining-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# Install Ollama and pull model (for RAG)
# See https://ollama.ai/download
ollama pull mistral
```

### Hardware Requirements
- **RAM**: 8 GB minimum (16 GB recommended for KGE training)
- **Disk**: ~2 GB for models and data
- **GPU**: Optional (CPU works for this project scale)

## How to Run

### Full Pipeline
```bash
python run_pipeline.py --step all
```

### Individual Steps
```bash
# Step 1: Crawl Wikipedia AI articles
python run_pipeline.py --step crawl

# Step 2: Information Extraction (NER + relations)
python run_pipeline.py --step ie

# Step 3: Build Knowledge Graph + Ontology + Alignment + Expansion
python run_pipeline.py --step kg

# Step 4: SWRL Reasoning demo
python run_pipeline.py --step reason

# Step 5: Knowledge Graph Embeddings
python run_pipeline.py --step kge

# Step 6: RAG evaluation
python run_pipeline.py --step rag
```

### Run without crawling (use pre-crawled data)
```bash
python run_pipeline.py --step all --skip-crawl
```

### Interactive RAG Demo
```bash
python -m src.rag.rag_pipeline
```
Then type questions like:
- "Who developed GPT-4?"
- "What field does the Transformer belong to?"
- "Which organization is Geoffrey Hinton affiliated with?"

## Demo Screenshot

![RAG Demo](docs/demo_screenshot.png)

*Interactive assistant answering questions grounded in the Knowledge Graph*

## Key Results

| Model   | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------|-------|--------|--------|---------|
| TransE  | 0.009 | 0.001  | 0.009  | 0.025   |
| ComplEx | 0.031 | 0.019  | 0.034  | 0.054   |

> ComplEx outperforms TransE across all metrics. Scores are low due to the high entity-to-triple ratio (~23K entities for ~25K triples), which is typical for large, sparse KGs. See Section 4.4 of the report for size sensitivity analysis.

## Domain

**Artificial Intelligence & Machine Learning**: The KB covers AI researchers, organizations, algorithms, models, frameworks, datasets, and subfields extracted from 60+ Wikipedia articles.

## Tools & Libraries

- **Crawling**: requests, trafilatura, BeautifulSoup4
- **NLP/NER**: spaCy (en_core_web_sm)
- **Knowledge Graph**: rdflib (RDF/OWL/SPARQL)
- **Reasoning**: OWLReady2 (SWRL rules)
- **Embeddings**: PyKEEN (TransE, ComplEx)
- **RAG/LLM**: Ollama (Mistral)

## Authors

- Bilal Bamba — ESILV 2026
- Maximilien Aired — ESILV 2026

## License

MIT License
