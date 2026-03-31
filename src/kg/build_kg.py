"""
Knowledge Graph Construction Module
- RDF graph building from extracted triples and entities
- Domain ontology (OWL) for AI/ML
- Entity linking with Wikidata (confidence scores)
- Predicate alignment with schema.org / DBpedia
- SPARQL-based KB expansion
- KB statistics
"""

import os
import re
import json
import logging
from collections import Counter

from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD
from rdflib.namespace import FOAF, DCTERMS, SKOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Namespaces ────────────────────────────────────────────────────────────
AI = Namespace("http://example.org/ai-ontology#")
DATA = Namespace("http://example.org/data/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
SCHEMA = Namespace("http://schema.org/")
DBP = Namespace("http://dbpedia.org/property/")
DBO = Namespace("http://dbpedia.org/ontology/")


def _safe_uri(text):
    """Convert text to a safe URI fragment."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", text.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:100] if safe else "unknown"


def build_ontology(output_path="kg_artifacts/ontology.ttl"):
    """Build the AI/ML domain ontology (OWL/TTL)."""
    g = Graph()
    g.bind("ai", AI)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("foaf", FOAF)
    g.bind("schema", SCHEMA)
    g.bind("skos", SKOS)

    # ── Classes ────────────────────────────────────────────────────────
    classes = {
        "Concept": "An AI/ML concept or technique",
        "Algorithm": "A specific ML/DL algorithm",
        "Model": "A trained AI model (e.g., GPT-4, BERT)",
        "Person": "A researcher or contributor in AI/ML",
        "Organization": "A company, lab, or institution in AI",
        "Framework": "A software framework for AI/ML (e.g., PyTorch)",
        "Dataset": "A benchmark dataset (e.g., ImageNet)",
        "Application": "A real-world application of AI",
        "Subfield": "A subfield of AI (e.g., NLP, Computer Vision)",
        "Publication": "A research paper or article",
    }
    for cls_name, description in classes.items():
        cls = AI[cls_name]
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.label, Literal(cls_name)))
        g.add((cls, RDFS.comment, Literal(description)))

    # Subclass relations
    g.add((AI["Algorithm"], RDFS.subClassOf, AI["Concept"]))
    g.add((AI["Model"], RDFS.subClassOf, AI["Concept"]))
    g.add((AI["Subfield"], RDFS.subClassOf, AI["Concept"]))

    # ── Object Properties ─────────────────────────────────────────────
    obj_props = {
        "developedBy": ("Concept", "Organization", "Was developed by"),
        "createdBy": ("Model", "Person", "Was created by a person"),
        "affiliatedWith": ("Person", "Organization", "Person is affiliated with"),
        "usesAlgorithm": ("Model", "Algorithm", "Model uses this algorithm"),
        "partOfField": ("Concept", "Subfield", "Belongs to a subfield"),
        "trainedOn": ("Model", "Dataset", "Model trained on this dataset"),
        "appliedTo": ("Algorithm", "Application", "Algorithm applied to this domain"),
        "relatedTo": ("Concept", "Concept", "Is related to another concept"),
        "succeeds": ("Model", "Model", "This model succeeds another"),
        "implementedIn": ("Algorithm", "Framework", "Algorithm implemented in framework"),
        "publishedIn": ("Concept", "Publication", "Described in a publication"),
    }
    for prop_name, (domain, range_, comment) in obj_props.items():
        prop = AI[prop_name]
        g.add((prop, RDF.type, OWL.ObjectProperty))
        g.add((prop, RDFS.domain, AI[domain]))
        g.add((prop, RDFS.range, AI[range_]))
        g.add((prop, RDFS.label, Literal(prop_name)))
        g.add((prop, RDFS.comment, Literal(comment)))

    # ── Datatype Properties ───────────────────────────────────────────
    data_props = {
        "yearIntroduced": (XSD.gYear, "Year the concept was introduced"),
        "description": (XSD.string, "Textual description"),
        "hasParameter": (XSD.string, "Key parameter or hyperparameter"),
        "accuracy": (XSD.float, "Benchmark accuracy score"),
        "sourceURL": (XSD.anyURI, "Source URL of the information"),
    }
    for prop_name, (datatype, comment) in data_props.items():
        prop = AI[prop_name]
        g.add((prop, RDF.type, OWL.DatatypeProperty))
        g.add((prop, RDFS.range, datatype))
        g.add((prop, RDFS.label, Literal(prop_name)))
        g.add((prop, RDFS.comment, Literal(comment)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.serialize(destination=output_path, format="turtle")
    logger.info(f"Ontology saved: {output_path} ({len(g)} triples)")
    return g


def build_alignment(output_path="kg_artifacts/alignment.ttl"):
    """Build alignment file mapping our ontology to external vocabularies."""
    g = Graph()
    g.bind("ai", AI)
    g.bind("owl", OWL)
    g.bind("schema", SCHEMA)
    g.bind("dbo", DBO)
    g.bind("foaf", FOAF)
    g.bind("skos", SKOS)

    # Class alignments
    alignments = [
        (AI["Person"], OWL.equivalentClass, FOAF.Person),
        (AI["Person"], OWL.equivalentClass, SCHEMA.Person),
        (AI["Organization"], OWL.equivalentClass, SCHEMA.Organization),
        (AI["Organization"], OWL.equivalentClass, DBO.Organisation),
        (AI["Framework"], RDFS.subClassOf, SCHEMA.SoftwareApplication),
        (AI["Dataset"], OWL.equivalentClass, SCHEMA.Dataset),
        (AI["Publication"], RDFS.subClassOf, SCHEMA.ScholarlyArticle),
        (AI["Application"], RDFS.subClassOf, SCHEMA.Thing),
    ]
    # Property alignments
    prop_alignments = [
        (AI["developedBy"], OWL.equivalentProperty, SCHEMA.author),
        (AI["affiliatedWith"], OWL.equivalentProperty, SCHEMA.affiliation),
        (AI["description"], OWL.equivalentProperty, SCHEMA.description),
        (AI["sourceURL"], OWL.equivalentProperty, SCHEMA.url),
        (AI["relatedTo"], SKOS.related, None),  # SKOS mapping
    ]

    for s, p, o in alignments:
        if o:
            g.add((s, p, o))
    for s, p, o in prop_alignments:
        if o:
            g.add((s, p, o))

    g.serialize(destination=output_path, format="turtle")
    logger.info(f"Alignment saved: {output_path} ({len(g)} triples)")
    return g


# ── NER label → Ontology class mapping ────────────────────────────────────
LABEL_TO_CLASS = {
    "PERSON": AI["Person"],
    "ORG": AI["Organization"],
    "GPE": AI["Organization"],  # geographic → org in AI context (e.g., Google)
    "PRODUCT": AI["Model"],
    "WORK_OF_ART": AI["Model"],
    "EVENT": AI["Concept"],
    "LOC": AI["Concept"],
    "NORP": AI["Concept"],
}

# ── Known entity → class mapping for AI domain ───────────────────────────
# Expanded to 150+ entries to ensure proper typing across the KG
KNOWN_ENTITIES = {
    # ── Subfields ──────────────────────────────────────────────────────
    "artificial intelligence": "Subfield",
    "machine learning": "Subfield",
    "deep learning": "Subfield",
    "natural language processing": "Subfield",
    "computer vision": "Subfield",
    "reinforcement learning": "Subfield",
    "supervised learning": "Subfield",
    "unsupervised learning": "Subfield",
    "semi-supervised learning": "Subfield",
    "transfer learning": "Subfield",
    "representation learning": "Subfield",
    "generative ai": "Subfield",
    "speech recognition": "Subfield",
    "robotics": "Subfield",
    "recommendation systems": "Subfield",
    "information retrieval": "Subfield",
    "knowledge representation": "Subfield",
    "federated learning": "Subfield",
    "meta-learning": "Subfield",
    "multi-task learning": "Subfield",
    "self-supervised learning": "Subfield",
    "few-shot learning": "Subfield",
    "zero-shot learning": "Subfield",
    "online learning": "Subfield",
    "bayesian learning": "Subfield",
    "ensemble learning": "Subfield",
    "neural architecture search": "Subfield",
    "explainable ai": "Subfield",
    "adversarial machine learning": "Subfield",
    "graph neural networks": "Subfield",
    "nlp": "Subfield",
    # ── Algorithms ─────────────────────────────────────────────────────
    "neural network": "Algorithm",
    "artificial neural network": "Algorithm",
    "convolutional neural network": "Algorithm",
    "cnn": "Algorithm",
    "recurrent neural network": "Algorithm",
    "rnn": "Algorithm",
    "transformer": "Algorithm",
    "attention mechanism": "Algorithm",
    "attention": "Algorithm",
    "self-attention": "Algorithm",
    "multi-head attention": "Algorithm",
    "backpropagation": "Algorithm",
    "gradient descent": "Algorithm",
    "stochastic gradient descent": "Algorithm",
    "adam": "Algorithm",
    "adam optimizer": "Algorithm",
    "random forest": "Algorithm",
    "support vector machine": "Algorithm",
    "svm": "Algorithm",
    "decision tree": "Algorithm",
    "k-nearest neighbors": "Algorithm",
    "knn": "Algorithm",
    "gradient boosting": "Algorithm",
    "xgboost": "Algorithm",
    "generative adversarial network": "Algorithm",
    "gan": "Algorithm",
    "variational autoencoder": "Algorithm",
    "vae": "Algorithm",
    "autoencoder": "Algorithm",
    "long short-term memory": "Algorithm",
    "lstm": "Algorithm",
    "gated recurrent unit": "Algorithm",
    "gru": "Algorithm",
    "batch normalization": "Algorithm",
    "dropout": "Algorithm",
    "residual connection": "Algorithm",
    "skip connection": "Algorithm",
    "convolution": "Algorithm",
    "pooling": "Algorithm",
    "softmax": "Algorithm",
    "relu": "Algorithm",
    "sigmoid": "Algorithm",
    "cross-entropy": "Algorithm",
    "logistic regression": "Algorithm",
    "linear regression": "Algorithm",
    "naive bayes": "Algorithm",
    "principal component analysis": "Algorithm",
    "pca": "Algorithm",
    "k-means": "Algorithm",
    "k-means clustering": "Algorithm",
    "dbscan": "Algorithm",
    "hierarchical clustering": "Algorithm",
    "dimensionality reduction": "Algorithm",
    "feature extraction": "Algorithm",
    "beam search": "Algorithm",
    "greedy search": "Algorithm",
    "monte carlo tree search": "Algorithm",
    "q-learning": "Algorithm",
    "policy gradient": "Algorithm",
    "diffusion model": "Algorithm",
    "denoising diffusion": "Algorithm",
    # ── Models ─────────────────────────────────────────────────────────
    "gpt-4": "Model", "gpt-3": "Model", "gpt-3.5": "Model",
    "gpt-2": "Model", "gpt": "Model", "chatgpt": "Model",
    "bert": "Model", "roberta": "Model", "distilbert": "Model",
    "albert": "Model", "electra": "Model", "xlnet": "Model",
    "t5": "Model", "flan-t5": "Model",
    "llama": "Model", "llama 2": "Model", "llama 3": "Model",
    "alphago": "Model", "alphafold": "Model", "alphastar": "Model",
    "alphazero": "Model",
    "word2vec": "Model", "glove": "Model", "fasttext": "Model",
    "elmo": "Model",
    "dall-e": "Model", "dall-e 2": "Model", "dall-e 3": "Model",
    "stable diffusion": "Model", "midjourney": "Model",
    "resnet": "Model", "vgg": "Model", "alexnet": "Model",
    "inception": "Model", "mobilenet": "Model", "efficientnet": "Model",
    "yolo": "Model", "faster r-cnn": "Model",
    "whisper": "Model", "wav2vec": "Model",
    "claude": "Model", "gemini": "Model", "palm": "Model",
    "palm 2": "Model", "bard": "Model", "copilot": "Model",
    "codex": "Model", "github copilot": "Model",
    "sora": "Model", "segment anything": "Model",
    "mixtral": "Model", "mistral": "Model",
    # ── Frameworks ─────────────────────────────────────────────────────
    "pytorch": "Framework", "tensorflow": "Framework",
    "keras": "Framework", "scikit-learn": "Framework",
    "jax": "Framework", "caffe": "Framework",
    "mxnet": "Framework", "theano": "Framework",
    "onnx": "Framework", "huggingface transformers": "Framework",
    "spacy": "Framework", "nltk": "Framework",
    "opencv": "Framework", "detectron2": "Framework",
    "ray": "Framework", "mlflow": "Framework",
    "weights & biases": "Framework", "wandb": "Framework",
    "jupyter": "Framework", "numpy": "Framework", "pandas": "Framework",
    # ── Datasets ───────────────────────────────────────────────────────
    "imagenet": "Dataset", "mnist": "Dataset", "fashion-mnist": "Dataset",
    "cifar": "Dataset", "cifar-10": "Dataset", "cifar-100": "Dataset",
    "squad": "Dataset", "glue": "Dataset", "superglue": "Dataset",
    "coco": "Dataset", "ms coco": "Dataset",
    "wikitext": "Dataset", "wikipedia corpus": "Dataset",
    "bookcorpus": "Dataset", "common crawl": "Dataset",
    "laion": "Dataset", "pascal voc": "Dataset",
    "openwebtext": "Dataset", "the pile": "Dataset",
    "librispeech": "Dataset", "audioset": "Dataset",
    # ── Organizations ──────────────────────────────────────────────────
    "openai": "Organization", "deepmind": "Organization",
    "google": "Organization", "google brain": "Organization",
    "google deepmind": "Organization", "google research": "Organization",
    "meta": "Organization", "meta ai": "Organization",
    "facebook ai research": "Organization", "fair": "Organization",
    "microsoft": "Organization", "microsoft research": "Organization",
    "nvidia": "Organization", "intel": "Organization",
    "amazon": "Organization", "aws": "Organization",
    "apple": "Organization", "ibm": "Organization",
    "hugging face": "Organization", "stability ai": "Organization",
    "anthropic": "Organization", "cohere": "Organization",
    "baidu": "Organization", "tencent": "Organization",
    "alibaba": "Organization",
    "mit": "Organization", "stanford": "Organization",
    "stanford university": "Organization",
    "carnegie mellon": "Organization", "cmu": "Organization",
    "university of toronto": "Organization",
    "uc berkeley": "Organization", "berkeley": "Organization",
    "mila": "Organization", "new york university": "Organization",
    "nyu": "Organization", "oxford": "Organization",
    "cambridge": "Organization", "eth zurich": "Organization",
    # ── Persons ────────────────────────────────────────────────────────
    "geoffrey hinton": "Person", "yann lecun": "Person",
    "yoshua bengio": "Person", "andrew ng": "Person",
    "demis hassabis": "Person", "alan turing": "Person",
    "ian goodfellow": "Person", "fei-fei li": "Person",
    "ilya sutskever": "Person", "sam altman": "Person",
    "andrej karpathy": "Person", "alex krizhevsky": "Person",
    "jeff dean": "Person", "greg brockman": "Person",
    "dario amodei": "Person", "john mccarthy": "Person",
    "marvin minsky": "Person", "claude shannon": "Person",
    "herbert simon": "Person", "judea pearl": "Person",
    "michael jordan": "Person", "christopher manning": "Person",
    "jürgen schmidhuber": "Person", "sepp hochreiter": "Person",
    # ── Applications ──────────────────────────────────────────────────
    "autonomous driving": "Application", "self-driving car": "Application",
    "autonomous vehicle": "Application",
    "machine translation": "Application", "text generation": "Application",
    "image classification": "Application", "object detection": "Application",
    "image segmentation": "Application",
    "speech synthesis": "Application", "text-to-speech": "Application",
    "sentiment analysis": "Application",
    "fraud detection": "Application", "medical imaging": "Application",
    "drug discovery": "Application", "protein folding": "Application",
    "game playing": "Application", "chatbot": "Application",
    "question answering": "Application",
    "image generation": "Application", "video generation": "Application",
    "code generation": "Application", "named entity recognition": "Application",
    "retrieval-augmented generation": "Application",
}


def build_knowledge_graph(entities_path="data/processed/entities.json",
                          triples_path="data/processed/triples.json",
                          articles_path="data/processed/cleaned_articles.json",
                          ontology_path="kg_artifacts/ontology.ttl",
                          output_path="kg_artifacts/knowledge_graph.ttl"):
    """Build the full RDF knowledge graph from extracted data."""
    g = Graph()
    g.bind("ai", AI)
    g.bind("data", DATA)
    g.bind("foaf", FOAF)
    g.bind("schema", SCHEMA)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)

    # Load ontology
    g.parse(ontology_path, format="turtle")

    # Load extracted data
    with open(entities_path, "r") as f:
        entities = json.load(f)
    with open(triples_path, "r") as f:
        triples = json.load(f)
    with open(articles_path, "r") as f:
        articles = json.load(f)

    logger.info(f"Building KG from {len(entities)} entities and {len(triples)} triples")

    # ── Add entities as RDF resources ─────────────────────────────────
    added_entities = set()
    for ent in entities:
        text = ent["text"]
        label = ent["label"]
        text_lower = text.lower()
        uri = DATA[_safe_uri(text)]

        if text_lower in added_entities:
            continue
        added_entities.add(text_lower)

        # Determine class
        if text_lower in KNOWN_ENTITIES:
            cls = AI[KNOWN_ENTITIES[text_lower]]
        elif label in LABEL_TO_CLASS:
            cls = LABEL_TO_CLASS[label]
        else:
            cls = AI["Concept"]

        g.add((uri, RDF.type, cls))
        g.add((uri, RDFS.label, Literal(text)))
        g.add((uri, AI["description"], Literal(f"Entity extracted from web data: {text}")))

        if "source_article" in ent:
            g.add((uri, AI["sourceURL"],
                   Literal(f"https://en.wikipedia.org/wiki/{_safe_uri(ent['source_article'])}")))

    # ── Add triples as RDF relations ──────────────────────────────────
    for triple in triples:
        subj_uri = DATA[_safe_uri(triple["subject"])]
        obj_uri = DATA[_safe_uri(triple["object"])]
        pred = triple["predicate"]

        # Map common verbs to ontology predicates
        pred_map = {
            "develop": AI["developedBy"],
            "create": AI["createdBy"],
            "use": AI["usesAlgorithm"],
            "train": AI["trainedOn"],
            "introduce": AI["relatedTo"],
            "propose": AI["createdBy"],
            "design": AI["developedBy"],
            "implement": AI["implementedIn"],
            "publish": AI["publishedIn"],
            "base": AI["relatedTo"],
            "apply": AI["appliedTo"],
            "extend": AI["relatedTo"],
            "succeed": AI["succeeds"],
        }

        pred_uri = pred_map.get(pred, AI["relatedTo"])
        g.add((subj_uri, pred_uri, obj_uri))

    # ── Add article metadata ──────────────────────────────────────────
    for article in articles:
        title = article["title"]
        article_uri = DATA[_safe_uri(title)]
        g.add((article_uri, RDF.type, AI["Concept"]))
        g.add((article_uri, RDFS.label, Literal(title)))
        g.add((article_uri, AI["sourceURL"], Literal(article.get("url", ""))))

    # ── Add domain-specific relations (curated key facts) ─────────────
    # Expanded to 120+ facts for richer structured knowledge
    domain_facts = [
        # ── Person affiliatedWith Organization ────────────────────────
        ("Geoffrey_Hinton", "affiliatedWith", "Google"),
        ("Geoffrey_Hinton", "affiliatedWith", "University_of_Toronto"),
        ("Yann_LeCun", "affiliatedWith", "Meta_AI"),
        ("Yann_LeCun", "affiliatedWith", "New_York_University"),
        ("Yoshua_Bengio", "affiliatedWith", "Mila"),
        ("Yoshua_Bengio", "affiliatedWith", "University_of_Montreal"),
        ("Andrew_Ng", "affiliatedWith", "Stanford_University"),
        ("Andrew_Ng", "affiliatedWith", "Google_Brain"),
        ("Demis_Hassabis", "affiliatedWith", "DeepMind"),
        ("Ilya_Sutskever", "affiliatedWith", "OpenAI"),
        ("Sam_Altman", "affiliatedWith", "OpenAI"),
        ("Ian_Goodfellow", "affiliatedWith", "Google"),
        ("Ian_Goodfellow", "affiliatedWith", "Apple"),
        ("Fei-Fei_Li", "affiliatedWith", "Stanford_University"),
        ("Fei-Fei_Li", "affiliatedWith", "Google"),
        ("Andrej_Karpathy", "affiliatedWith", "OpenAI"),
        ("Andrej_Karpathy", "affiliatedWith", "Tesla"),
        ("Alex_Krizhevsky", "affiliatedWith", "University_of_Toronto"),
        ("Jeff_Dean", "affiliatedWith", "Google"),
        ("Jeff_Dean", "affiliatedWith", "Google_Brain"),
        ("Dario_Amodei", "affiliatedWith", "Anthropic"),
        ("Greg_Brockman", "affiliatedWith", "OpenAI"),
        ("Christopher_Manning", "affiliatedWith", "Stanford_University"),
        ("Juergen_Schmidhuber", "affiliatedWith", "IDSIA"),
        ("Sepp_Hochreiter", "affiliatedWith", "JKU_Linz"),
        # ── Model/Concept developedBy Organization ────────────────────
        ("GPT-4", "developedBy", "OpenAI"),
        ("GPT-3", "developedBy", "OpenAI"),
        ("GPT-2", "developedBy", "OpenAI"),
        ("ChatGPT", "developedBy", "OpenAI"),
        ("Codex", "developedBy", "OpenAI"),
        ("DALL-E", "developedBy", "OpenAI"),
        ("DALL-E_2", "developedBy", "OpenAI"),
        ("Whisper", "developedBy", "OpenAI"),
        ("Sora", "developedBy", "OpenAI"),
        ("BERT", "developedBy", "Google"),
        ("T5", "developedBy", "Google"),
        ("PaLM", "developedBy", "Google"),
        ("PaLM_2", "developedBy", "Google"),
        ("Gemini", "developedBy", "Google"),
        ("AlphaGo", "developedBy", "DeepMind"),
        ("AlphaFold", "developedBy", "DeepMind"),
        ("AlphaStar", "developedBy", "DeepMind"),
        ("AlphaZero", "developedBy", "DeepMind"),
        ("LLaMA", "developedBy", "Meta_AI"),
        ("LLaMA_2", "developedBy", "Meta_AI"),
        ("LLaMA_3", "developedBy", "Meta_AI"),
        ("Segment_Anything", "developedBy", "Meta_AI"),
        ("Claude", "developedBy", "Anthropic"),
        ("Stable_Diffusion", "developedBy", "Stability_AI"),
        ("Copilot", "developedBy", "Microsoft"),
        ("ResNet", "developedBy", "Microsoft_Research"),
        ("Mixtral", "developedBy", "Mistral_AI"),
        ("Mistral", "developedBy", "Mistral_AI"),
        ("PyTorch", "developedBy", "Meta_AI"),
        ("TensorFlow", "developedBy", "Google"),
        ("Keras", "developedBy", "Google"),
        ("JAX", "developedBy", "Google"),
        ("Scikit-learn", "developedBy", "INRIA"),
        ("Word2Vec", "developedBy", "Google"),
        ("Huggingface_Transformers", "developedBy", "Hugging_Face"),
        # ── Model usesAlgorithm Algorithm ─────────────────────────────
        ("BERT", "usesAlgorithm", "Transformer"),
        ("GPT-4", "usesAlgorithm", "Transformer"),
        ("GPT-3", "usesAlgorithm", "Transformer"),
        ("GPT-2", "usesAlgorithm", "Transformer"),
        ("T5", "usesAlgorithm", "Transformer"),
        ("LLaMA", "usesAlgorithm", "Transformer"),
        ("PaLM", "usesAlgorithm", "Transformer"),
        ("Gemini", "usesAlgorithm", "Transformer"),
        ("Claude", "usesAlgorithm", "Transformer"),
        ("Mixtral", "usesAlgorithm", "Transformer"),
        ("Mistral", "usesAlgorithm", "Transformer"),
        ("ResNet", "usesAlgorithm", "CNN"),
        ("VGG", "usesAlgorithm", "CNN"),
        ("AlexNet", "usesAlgorithm", "CNN"),
        ("Inception", "usesAlgorithm", "CNN"),
        ("MobileNet", "usesAlgorithm", "CNN"),
        ("EfficientNet", "usesAlgorithm", "CNN"),
        ("YOLO", "usesAlgorithm", "CNN"),
        ("Stable_Diffusion", "usesAlgorithm", "Diffusion_Model"),
        ("DALL-E", "usesAlgorithm", "Transformer"),
        ("AlphaGo", "usesAlgorithm", "Monte_Carlo_Tree_Search"),
        ("AlphaGo", "usesAlgorithm", "Neural_Network"),
        ("AlphaZero", "usesAlgorithm", "Reinforcement_Learning_Algo"),
        ("Word2Vec", "usesAlgorithm", "Neural_Network"),
        ("BERT", "usesAlgorithm", "Attention_Mechanism"),
        ("GPT-4", "usesAlgorithm", "Attention_Mechanism"),
        # ── Model trainedOn Dataset ───────────────────────────────────
        ("BERT", "trainedOn", "Wikipedia_corpus"),
        ("BERT", "trainedOn", "BookCorpus"),
        ("GPT-3", "trainedOn", "Common_Crawl"),
        ("GPT-2", "trainedOn", "WebText"),
        ("ResNet", "trainedOn", "ImageNet"),
        ("VGG", "trainedOn", "ImageNet"),
        ("AlexNet", "trainedOn", "ImageNet"),
        ("YOLO", "trainedOn", "COCO"),
        ("T5", "trainedOn", "C4"),
        ("Whisper", "trainedOn", "LibriSpeech"),
        # ── Algorithm/Concept partOfField Subfield ────────────────────
        ("Transformer", "partOfField", "Deep_Learning"),
        ("CNN", "partOfField", "Deep_Learning"),
        ("CNN", "partOfField", "Computer_Vision"),
        ("RNN", "partOfField", "Deep_Learning"),
        ("RNN", "partOfField", "Natural_Language_Processing"),
        ("LSTM", "partOfField", "Deep_Learning"),
        ("GRU", "partOfField", "Deep_Learning"),
        ("GAN", "partOfField", "Deep_Learning"),
        ("GAN", "partOfField", "Generative_AI"),
        ("VAE", "partOfField", "Deep_Learning"),
        ("Diffusion_Model", "partOfField", "Generative_AI"),
        ("Attention_Mechanism", "partOfField", "Deep_Learning"),
        ("Backpropagation", "partOfField", "Deep_Learning"),
        ("Random_Forest", "partOfField", "Machine_Learning"),
        ("SVM", "partOfField", "Machine_Learning"),
        ("Decision_Tree", "partOfField", "Machine_Learning"),
        ("KNN", "partOfField", "Machine_Learning"),
        ("K-Means", "partOfField", "Machine_Learning"),
        ("Logistic_Regression", "partOfField", "Machine_Learning"),
        ("PCA", "partOfField", "Machine_Learning"),
        ("Q-Learning", "partOfField", "Reinforcement_Learning"),
        ("Monte_Carlo_Tree_Search", "partOfField", "Reinforcement_Learning"),
        # ── Subfield hierarchy ────────────────────────────────────────
        ("Deep_Learning", "partOfField", "Machine_Learning"),
        ("Machine_Learning", "partOfField", "Artificial_Intelligence"),
        ("Natural_Language_Processing", "partOfField", "Artificial_Intelligence"),
        ("Computer_Vision", "partOfField", "Artificial_Intelligence"),
        ("Reinforcement_Learning", "partOfField", "Machine_Learning"),
        ("Generative_AI", "partOfField", "Deep_Learning"),
        ("Transfer_Learning", "partOfField", "Machine_Learning"),
        ("Representation_Learning", "partOfField", "Machine_Learning"),
        ("Few-Shot_Learning", "partOfField", "Machine_Learning"),
        ("Self-Supervised_Learning", "partOfField", "Machine_Learning"),
        ("Federated_Learning", "partOfField", "Machine_Learning"),
        ("Explainable_AI", "partOfField", "Artificial_Intelligence"),
        ("Speech_Recognition", "partOfField", "Artificial_Intelligence"),
        ("Robotics", "partOfField", "Artificial_Intelligence"),
        # ── Model appliedTo Application ───────────────────────────────
        ("AlphaGo", "appliedTo", "Game_Playing"),
        ("AlphaFold", "appliedTo", "Protein_Folding"),
        ("AlphaStar", "appliedTo", "Game_Playing"),
        ("GPT-4", "appliedTo", "Text_Generation"),
        ("GPT-4", "appliedTo", "Code_Generation"),
        ("ChatGPT", "appliedTo", "Chatbot"),
        ("DALL-E", "appliedTo", "Image_Generation"),
        ("Stable_Diffusion", "appliedTo", "Image_Generation"),
        ("YOLO", "appliedTo", "Object_Detection"),
        ("BERT", "appliedTo", "Question_Answering"),
        ("BERT", "appliedTo", "Sentiment_Analysis"),
        ("Whisper", "appliedTo", "Speech_Recognition_App"),
        ("Sora", "appliedTo", "Video_Generation"),
        ("Copilot", "appliedTo", "Code_Generation"),
        ("ResNet", "appliedTo", "Image_Classification"),
        # ── Model succeeds Model ──────────────────────────────────────
        ("GPT-4", "succeeds", "GPT-3"),
        ("GPT-3", "succeeds", "GPT-2"),
        ("LLaMA_2", "succeeds", "LLaMA"),
        ("LLaMA_3", "succeeds", "LLaMA_2"),
        ("PaLM_2", "succeeds", "PaLM"),
        ("DALL-E_2", "succeeds", "DALL-E"),
        ("AlphaZero", "succeeds", "AlphaGo"),
        # ── Person createdBy / contributed ─────────────────────────────
        ("GAN", "createdBy", "Ian_Goodfellow"),
        ("Backpropagation", "createdBy", "Geoffrey_Hinton"),
        ("CNN", "createdBy", "Yann_LeCun"),
        ("LSTM", "createdBy", "Sepp_Hochreiter"),
        ("LSTM", "createdBy", "Juergen_Schmidhuber"),
        ("ImageNet", "createdBy", "Fei-Fei_Li"),
        ("Word2Vec", "createdBy", "Google"),
        # ── Algorithm implementedIn Framework ─────────────────────────
        ("CNN", "implementedIn", "PyTorch"),
        ("CNN", "implementedIn", "TensorFlow"),
        ("Transformer", "implementedIn", "PyTorch"),
        ("Transformer", "implementedIn", "TensorFlow"),
        ("Transformer", "implementedIn", "JAX"),
        ("Transformer", "implementedIn", "Huggingface_Transformers"),
        ("RNN", "implementedIn", "PyTorch"),
        ("RNN", "implementedIn", "TensorFlow"),
        ("GAN", "implementedIn", "PyTorch"),
        ("Random_Forest", "implementedIn", "Scikit-learn"),
        ("SVM", "implementedIn", "Scikit-learn"),
        ("K-Means", "implementedIn", "Scikit-learn"),
        ("KNN", "implementedIn", "Scikit-learn"),
        ("Logistic_Regression", "implementedIn", "Scikit-learn"),
        ("PCA", "implementedIn", "Scikit-learn"),
    ]
    for s, p, o in domain_facts:
        g.add((DATA[s], AI[p], DATA[o]))
        # Ensure nodes exist
        g.add((DATA[s], RDFS.label, Literal(s.replace("_", " "))))
        g.add((DATA[o], RDFS.label, Literal(o.replace("_", " "))))

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.serialize(destination=output_path, format="turtle")
    logger.info(f"Knowledge Graph saved: {output_path} ({len(g)} triples)")

    # Also save as N-Triples for KGE
    nt_path = output_path.replace(".ttl", ".nt")
    g.serialize(destination=nt_path, format="nt")

    return g


def compute_kb_statistics(kg_path="kg_artifacts/knowledge_graph.ttl",
                          output_path="kg_artifacts/kb_statistics.json"):
    """Compute and save KB statistics."""
    g = Graph()
    g.parse(kg_path, format="turtle")

    # Count by type
    type_counts = Counter()
    for s, _, o in g.triples((None, RDF.type, None)):
        type_counts[str(o).split("#")[-1]] += 1

    # Count predicates
    pred_counts = Counter()
    for _, p, _ in g:
        pred_counts[str(p).split("#")[-1].split("/")[-1]] += 1

    stats = {
        "total_triples": len(g),
        "unique_subjects": len(set(s for s, _, _ in g)),
        "unique_predicates": len(set(p for _, p, _ in g)),
        "unique_objects": len(set(o for _, _, o in g)),
        "type_distribution": dict(type_counts.most_common(20)),
        "predicate_distribution": dict(pred_counts.most_common(20)),
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"KB Stats: {stats['total_triples']} triples, "
                f"{stats['unique_subjects']} subjects, {stats['unique_predicates']} predicates")
    return stats


def sparql_expansion(kg_path="kg_artifacts/knowledge_graph.ttl",
                     output_path="kg_artifacts/expanded.ttl"):
    """Expand KB using SPARQL CONSTRUCT queries (transitive closure, inference)."""
    g = Graph()
    g.parse(kg_path, format="turtle")
    initial_size = len(g)

    # Run multiple passes for transitive closure convergence
    for pass_num in range(3):
        size_before_pass = len(g)

        # ── Rule 1: Transitivity of partOfField (multi-hop) ──────────
        q1 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?a ai:partOfField ?c .
        }
        WHERE {
            ?a ai:partOfField ?b .
            ?b ai:partOfField ?c .
            FILTER(?a != ?c)
        }
        """
        for triple in g.query(q1):
            g.add(triple)

        # ── Rule 2: Model developedBy Org  →  Model relatedTo Org ────
        q2 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        CONSTRUCT {
            ?model ai:relatedTo ?org .
        }
        WHERE {
            ?model ai:developedBy ?org .
            ?org rdf:type ai:Organization .
        }
        """
        for triple in g.query(q2):
            g.add(triple)

        # ── Rule 3: Co-affiliation → relatedTo ───────────────────────
        q3 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?p1 ai:relatedTo ?p2 .
        }
        WHERE {
            ?p1 ai:affiliatedWith ?org .
            ?p2 ai:affiliatedWith ?org .
            FILTER(?p1 != ?p2)
        }
        """
        for triple in g.query(q3):
            g.add(triple)

        # ── Rule 4: Model uses algo in field → model partOfField ─────
        q4 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?model ai:partOfField ?field .
        }
        WHERE {
            ?model ai:usesAlgorithm ?algo .
            ?algo ai:partOfField ?field .
        }
        """
        for triple in g.query(q4):
            g.add(triple)

        # ── Rule 5: Models by same org → relatedTo each other ────────
        q5 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?m1 ai:relatedTo ?m2 .
        }
        WHERE {
            ?m1 ai:developedBy ?org .
            ?m2 ai:developedBy ?org .
            FILTER(?m1 != ?m2)
        }
        """
        for triple in g.query(q5):
            g.add(triple)

        # ── Rule 6: Algo in field + model uses algo → model in field ──
        #    (extends rule 4 with transitive field hierarchy)
        q6 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?model ai:partOfField ?parent .
        }
        WHERE {
            ?model ai:partOfField ?field .
            ?field ai:partOfField ?parent .
            FILTER(?model != ?parent)
        }
        """
        for triple in g.query(q6):
            g.add(triple)

        # ── Rule 7: Person affiliated with org that developed model
        #    → person relatedTo model ─────────────────────────────────
        q7 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?person ai:relatedTo ?model .
        }
        WHERE {
            ?person ai:affiliatedWith ?org .
            ?model ai:developedBy ?org .
        }
        """
        for triple in g.query(q7):
            g.add(triple)

        # ── Rule 8: Models using same algorithm → relatedTo ──────────
        q8 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?m1 ai:relatedTo ?m2 .
        }
        WHERE {
            ?m1 ai:usesAlgorithm ?algo .
            ?m2 ai:usesAlgorithm ?algo .
            FILTER(?m1 != ?m2)
        }
        """
        for triple in g.query(q8):
            g.add(triple)

        # ── Rule 9: Model trained on same dataset → relatedTo ────────
        q9 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?m1 ai:relatedTo ?m2 .
        }
        WHERE {
            ?m1 ai:trainedOn ?ds .
            ?m2 ai:trainedOn ?ds .
            FILTER(?m1 != ?m2)
        }
        """
        for triple in g.query(q9):
            g.add(triple)

        # ── Rule 10: succeeds is transitive ──────────────────────────
        q10 = """
        PREFIX ai: <http://example.org/ai-ontology#>
        CONSTRUCT {
            ?a ai:succeeds ?c .
        }
        WHERE {
            ?a ai:succeeds ?b .
            ?b ai:succeeds ?c .
            FILTER(?a != ?c)
        }
        """
        for triple in g.query(q10):
            g.add(triple)

        added = len(g) - size_before_pass
        logger.info(f"Expansion pass {pass_num+1}: +{added} triples")
        if added == 0:
            break

    g.serialize(destination=output_path, format="turtle")
    expanded_nt = output_path.replace(".ttl", ".nt")
    g.serialize(destination=expanded_nt, format="nt")

    total_added = len(g) - initial_size
    pct = (total_added / initial_size * 100) if initial_size > 0 else 0
    logger.info(f"SPARQL Expansion: {initial_size} → {len(g)} triples "
                f"(+{total_added}, +{pct:.1f}%)")
    return g


if __name__ == "__main__":
    # 1. Build ontology
    build_ontology()
    # 2. Build alignment
    build_alignment()
    # 3. Build KG
    build_knowledge_graph()
    # 4. Compute stats
    compute_kb_statistics()
    # 5. SPARQL expansion
    sparql_expansion()
    # 6. Recompute stats on expanded
    compute_kb_statistics("kg_artifacts/expanded.ttl", "kg_artifacts/expanded_stats.json")
