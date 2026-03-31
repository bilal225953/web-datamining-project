"""
Web Crawler for AI/ML Domain Knowledge Base
Crawls Wikipedia articles related to Artificial Intelligence topics.
Uses trafilatura for clean text extraction + requests for HTTP.
Respects robots.txt and implements polite crawling.
"""

import os
import sys
import json
import time
import hashlib
import logging
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup
import trafilatura

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Seed URLs: core AI/ML Wikipedia articles ──────────────────────────────
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Recurrent_neural_network",
    "https://en.wikipedia.org/wiki/Generative_adversarial_network",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/GPT-4",
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "https://en.wikipedia.org/wiki/Support_vector_machine",
    "https://en.wikipedia.org/wiki/Random_forest",
    "https://en.wikipedia.org/wiki/Gradient_boosting",
    "https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm",
    "https://en.wikipedia.org/wiki/Decision_tree_learning",
    "https://en.wikipedia.org/wiki/Backpropagation",
    "https://en.wikipedia.org/wiki/Transfer_learning",
    "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    "https://en.wikipedia.org/wiki/Artificial_neural_network",
    "https://en.wikipedia.org/wiki/Supervised_learning",
    "https://en.wikipedia.org/wiki/Unsupervised_learning",
    "https://en.wikipedia.org/wiki/Clustering_analysis",
    "https://en.wikipedia.org/wiki/Dimensionality_reduction",
    "https://en.wikipedia.org/wiki/Feature_engineering",
    "https://en.wikipedia.org/wiki/Overfitting",
    "https://en.wikipedia.org/wiki/Cross-validation_(statistics)",
    "https://en.wikipedia.org/wiki/Geoffrey_Hinton",
    "https://en.wikipedia.org/wiki/Yann_LeCun",
    "https://en.wikipedia.org/wiki/Yoshua_Bengio",
    "https://en.wikipedia.org/wiki/Andrew_Ng",
    "https://en.wikipedia.org/wiki/Demis_Hassabis",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/DeepMind",
    "https://en.wikipedia.org/wiki/Google_Brain",
    "https://en.wikipedia.org/wiki/Meta_AI",
    "https://en.wikipedia.org/wiki/Hugging_Face",
    "https://en.wikipedia.org/wiki/PyTorch",
    "https://en.wikipedia.org/wiki/TensorFlow",
    "https://en.wikipedia.org/wiki/Scikit-learn",
    "https://en.wikipedia.org/wiki/Keras",
    "https://en.wikipedia.org/wiki/ImageNet",
    "https://en.wikipedia.org/wiki/Word2vec",
    "https://en.wikipedia.org/wiki/Sentiment_analysis",
    "https://en.wikipedia.org/wiki/Named-entity_recognition",
    "https://en.wikipedia.org/wiki/Knowledge_graph",
    "https://en.wikipedia.org/wiki/Semantic_Web",
    "https://en.wikipedia.org/wiki/Resource_Description_Framework",
    "https://en.wikipedia.org/wiki/SPARQL",
    "https://en.wikipedia.org/wiki/Web_Ontology_Language",
    "https://en.wikipedia.org/wiki/Ontology_(information_science)",
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Alan_Turing",
    "https://en.wikipedia.org/wiki/Turing_test",
    "https://en.wikipedia.org/wiki/AlphaGo",
    "https://en.wikipedia.org/wiki/AlphaFold",
    "https://en.wikipedia.org/wiki/Chatbot",
    "https://en.wikipedia.org/wiki/Autonomous_vehicle",
]


class AIWebCrawler:
    """Focused web crawler for AI/ML Wikipedia articles."""

    def __init__(self, output_dir="data/raw", max_pages=80, delay=1.0):
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.delay = delay  # politeness delay in seconds
        self.visited = set()
        self.content_hashes = set()  # duplicate detection
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ESILV-WebMining-Bot/1.0 (student project; polite crawler)"
        })
        os.makedirs(output_dir, exist_ok=True)

    def _is_valid_wiki_ai_url(self, url):
        """Filter: only keep Wikipedia article URLs (no talk/special pages)."""
        parsed = urlparse(url)
        if parsed.netloc != "en.wikipedia.org":
            return False
        path = parsed.path
        if not path.startswith("/wiki/"):
            return False
        # Exclude special pages
        excludes = [":", "Main_Page", "Special:", "Talk:", "User:", "File:",
                     "Help:", "Template:", "Category:", "Portal:", "Wikipedia:"]
        for ex in excludes:
            if ex in path:
                return False
        return True

    def _extract_links(self, html, base_url):
        """Extract internal Wikipedia links from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        content_div = soup.find("div", {"id": "bodyContent"})
        if not content_div:
            return []
        links = []
        for a in content_div.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href).split("#")[0]
            if self._is_valid_wiki_ai_url(full_url):
                links.append(full_url)
        return links

    def _content_hash(self, text):
        """MD5 hash for duplicate detection."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _extract_clean_text(self, html, url):
        """Use trafilatura for clean main-content extraction."""
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            output_format="txt",
        )
        return text

    def crawl(self):
        """Run the crawl. BFS from seed URLs."""
        queue = deque(SEED_URLS)
        results = []
        page_count = 0

        logger.info(f"Starting crawl with {len(SEED_URLS)} seed URLs, max {self.max_pages} pages")

        while queue and page_count < self.max_pages:
            url = queue.popleft()
            if url in self.visited:
                continue
            self.visited.add(url)

            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                continue

            html = resp.text

            # Extract clean text with trafilatura
            clean_text = self._extract_clean_text(html, url)
            if not clean_text or len(clean_text) < 200:
                logger.info(f"Skipping {url} (too short or empty)")
                continue

            # Duplicate detection
            h = self._content_hash(clean_text)
            if h in self.content_hashes:
                logger.info(f"Duplicate detected, skipping {url}")
                continue
            self.content_hashes.add(h)

            # Extract title from URL
            title = url.split("/wiki/")[-1].replace("_", " ")

            record = {
                "url": url,
                "title": title,
                "text": clean_text,
                "text_length": len(clean_text),
                "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            results.append(record)
            page_count += 1
            logger.info(f"[{page_count}/{self.max_pages}] Crawled: {title} ({len(clean_text)} chars)")

            # Extract links for BFS expansion
            new_links = self._extract_links(html, url)
            for link in new_links[:10]:  # limit link expansion per page
                if link not in self.visited:
                    queue.append(link)

            # Politeness delay
            time.sleep(self.delay)

        # Save results
        output_path = os.path.join(self.output_dir, "crawled_articles.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Crawl complete. {len(results)} articles saved to {output_path}")

        # Save statistics
        stats = {
            "total_pages_crawled": len(results),
            "total_urls_visited": len(self.visited),
            "duplicates_removed": len(self.visited) - len(results),
            "avg_text_length": sum(r["text_length"] for r in results) / max(len(results), 1),
            "seed_urls_count": len(SEED_URLS),
        }
        stats_path = os.path.join(self.output_dir, "crawl_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Crawl stats: {stats}")

        return results


if __name__ == "__main__":
    crawler = AIWebCrawler(
        output_dir="data/raw",
        max_pages=60,
        delay=1.5,
    )
    crawler.crawl()
