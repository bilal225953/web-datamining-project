"""
Main Pipeline Runner
Runs the full end-to-end pipeline: Crawl → IE → KG → Reasoning → KGE → RAG
"""

import os
import sys
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_step(name, func, *args, **kwargs):
    logger.info(f"\n{'='*60}\n  STEP: {name}\n{'='*60}")
    try:
        result = func(*args, **kwargs)
        logger.info(f"  ✓ {name} complete\n")
        return result
    except Exception as e:
        logger.error(f"  ✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Web Datamining & Semantics Pipeline")
    parser.add_argument("--step", choices=["crawl", "ie", "kg", "reason", "kge", "rag", "all"],
                        default="all", help="Which step to run")
    parser.add_argument("--skip-crawl", action="store_true",
                        help="Skip crawling (use existing data)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.step in ("crawl", "all") and not args.skip_crawl:
        from src.crawl.crawler import AIWebCrawler
        crawler = AIWebCrawler(output_dir="data/raw", max_pages=60, delay=1.5)
        run_step("Web Crawling", crawler.crawl)

    if args.step in ("ie", "all"):
        from src.ie.extraction import run_ie_pipeline
        run_step("Information Extraction", run_ie_pipeline)

    if args.step in ("kg", "all"):
        from src.kg.build_kg import (build_ontology, build_alignment,
                                      build_knowledge_graph, compute_kb_statistics,
                                      sparql_expansion)
        run_step("Build Ontology", build_ontology)
        run_step("Build Alignment", build_alignment)
        run_step("Build Knowledge Graph", build_knowledge_graph)
        run_step("Compute KB Statistics", compute_kb_statistics)
        run_step("SPARQL Expansion", sparql_expansion)
        run_step("Expanded KB Statistics", compute_kb_statistics,
                 "kg_artifacts/expanded.ttl", "kg_artifacts/expanded_stats.json")

    if args.step in ("reason", "all"):
        from src.reason.swrl_reasoning import demo_family_swrl, demo_ai_swrl
        run_step("SWRL Family Demo", demo_family_swrl)
        run_step("SWRL AI KB Demo", demo_ai_swrl)

    if args.step in ("kge", "all"):
        from src.kge.embeddings import prepare_kge_data, train_kge_models
        run_step("Prepare KGE Data", prepare_kge_data)
        run_step("Train KGE Models", train_kge_models)

    if args.step in ("rag", "all"):
        from src.rag.rag_pipeline import RAGAssistant, run_evaluation
        try:
            assistant = RAGAssistant()
            run_step("RAG Evaluation", run_evaluation, assistant)
        except Exception as e:
            logger.warning(f"RAG evaluation skipped (Ollama may not be running): {e}")

    logger.info("\n" + "="*60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
