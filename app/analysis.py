from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = set(ENGLISH_STOP_WORDS)
BIO_STOPWORDS = {
    "study",
    "studies",
    "results",
    "conclusion",
    "background",
    "objective",
    "patients",
    "patient",
    "using",
    "used",
    "analysis",
    "data",
    "methods",
    "disease",
    "treatment",
    "associated",
    "significant",
}


class Analyzer:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=3500,
        )

    def run(self, query: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        docs = [f"{p['title']}\n{p['abstract']}" for p in papers]
        if not docs:
            return {
                "query": query,
                "themes": [],
                "papers": [],
                "graph": {"nodes": [], "edges": []},
                "overview": "No papers found.",
            }

        matrix = self.vectorizer.fit_transform(docs)
        query_vec = self.vectorizer.transform([query])
        relevance = cosine_similarity(query_vec, matrix)[0]

        for idx, paper in enumerate(papers):
            paper["relevance"] = round(float(relevance[idx]), 4)
            paper["keywords"] = self.extract_keywords(f"{paper['title']} {paper['abstract']}")
            paper["summary"] = self.extractive_summary(paper["abstract"] or paper["title"])
            paper["entities"] = self.extract_entities(paper)

        ranked_papers = sorted(papers, key=lambda x: x["relevance"], reverse=True)
        themes = self.cluster_papers(ranked_papers, matrix)
        graph = self.build_graph(ranked_papers)
        overview = self.build_overview(query, themes, ranked_papers)

        return {
            "query": query,
            "overview": overview,
            "themes": themes,
            "papers": ranked_papers,
            "graph": graph,
        }

    def extract_keywords(self, text: str, top_n: int = 8) -> List[str]:
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=40)
            mat = vec.fit_transform([text])
            scores = zip(vec.get_feature_names_out(), mat.toarray()[0])
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            return [term for term, score in ranked[:top_n] if score > 0]
        except ValueError:
            return []

    def extractive_summary(self, abstract: str, max_sentences: int = 2) -> str:
        if not abstract:
            return "No abstract available."
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        if len(sentences) <= max_sentences:
            return abstract[:700]

        words = re.findall(r"[A-Za-z][A-Za-z\-]+", abstract.lower())
        counts = Counter(w for w in words if w not in STOPWORDS and w not in BIO_STOPWORDS)
        sent_scores = []
        for sent in sentences:
            tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", sent.lower())
            score = sum(counts[t] for t in tokens)
            length_penalty = math.sqrt(max(len(tokens), 1))
            sent_scores.append((sent, score / length_penalty if length_penalty else score))
        chosen = [s for s, _ in sorted(sent_scores, key=lambda x: x[1], reverse=True)[:max_sentences]]
        return " ".join(chosen)

    def extract_entities(self, paper: Dict[str, Any]) -> Dict[str, List[str]]:
        text = f"{paper['title']} {paper['abstract']}"
        entities = {
            "genes_or_biomarkers": self._regex_find_all(r"\b[A-Z0-9]{2,8}\b", text, max_items=6),
            "drugs_or_compounds": self._regex_find_all(
                r"\b(?:[A-Z][a-z]+(?:mab|nib|ciclib|parib)|aspirin|metformin|temozolomide|cisplatin|nivolumab|pembrolizumab)\b",
                text,
                max_items=6,
            ),
            "modalities": self._regex_find_all(
                r"\b(?:MRI|CT|PET|RNA-seq|single-cell|spatial transcriptomics|machine learning|deep learning|randomized trial|cohort)\b",
                text,
                max_items=6,
            ),
        }
        return entities

    def cluster_papers(self, papers: List[Dict[str, Any]], matrix) -> List[Dict[str, Any]]:
        n_docs = len(papers)
        if n_docs == 1:
            return [{"theme": "Single paper", "keywords": papers[0]["keywords"][:5], "paper_indices": [0]}]

        n_clusters = min(3, n_docs)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(matrix)

        grouped: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            grouped[int(label)].append(idx)

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        themes = []
        for label, idxs in grouped.items():
            centroid = km.cluster_centers_[label]
            top_terms = feature_names[np.argsort(centroid)[::-1][:5]].tolist()
            theme_name = self._theme_name(top_terms)
            themes.append({
                "theme": theme_name,
                "keywords": top_terms,
                "paper_indices": idxs,
                "paper_titles": [papers[i]["title"] for i in idxs[:4]],
            })
        themes.sort(key=lambda x: len(x["paper_indices"]), reverse=True)
        return themes

    def build_graph(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        G = nx.Graph()
        keyword_counts = Counter()

        for paper in papers[:10]:
            paper_node = f"PMID:{paper['pmid']}"
            G.add_node(paper_node, label=paper["title"][:55], kind="paper", url=paper["url"])
            for kw in paper["keywords"][:5]:
                if len(kw) < 4:
                    continue
                keyword_counts[kw] += 1
                G.add_node(kw, label=kw, kind="concept")
                G.add_edge(paper_node, kw)

        # prune concepts seen once to keep graph clean
        for node in list(G.nodes):
            if G.nodes[node].get("kind") == "concept" and keyword_counts[node] < 2:
                G.remove_node(node)

        pos = nx.spring_layout(G, seed=42, k=1.1)
        nodes = []
        for node, attrs in G.nodes(data=True):
            x, y = pos[node]
            nodes.append({
                "id": node,
                "label": attrs.get("label", node),
                "kind": attrs.get("kind", "concept"),
                "url": attrs.get("url"),
                "x": float(x),
                "y": float(y),
            })
        edges = [{"source": u, "target": v} for u, v in G.edges()]
        return {"nodes": nodes, "edges": edges}

    def build_overview(self, query: str, themes: List[Dict[str, Any]], papers: List[Dict[str, Any]]) -> str:
        top_papers = papers[:3]
        years = [p["year"] for p in papers if str(p.get("year", "")).isdigit()]
        date_span = f"{min(years)} to {max(years)}" if years else "mixed publication years"
        theme_line = ", ".join(t["theme"] for t in themes[:3]) if themes else "one main theme"
        titles = "; ".join(p["title"][:90] for p in top_papers)
        return (
            f"For '{query}', the strongest papers cluster around {theme_line}. "
            f"The current set covers {date_span}. "
            f"Most relevant examples include {titles}."
        )

    @staticmethod
    def _regex_find_all(pattern: str, text: str, max_items: int = 5) -> List[str]:
        found = []
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            value = match[0] if isinstance(match, tuple) else match
            if value not in found:
                found.append(value)
            if len(found) >= max_items:
                break
        return found

    @staticmethod
    def _theme_name(top_terms: List[str]) -> str:
        if not top_terms:
            return "Mixed literature"
        return " / ".join(term.title() for term in top_terms[:2])


class QASystem:
    def answer(self, question: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not papers:
            return {"answer": "I do not have papers loaded yet.", "citations": []}

        scored = []
        q_tokens = set(re.findall(r"[A-Za-z][A-Za-z\-]+", question.lower()))
        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}".lower()
            score = sum(1 for token in q_tokens if token in text)
            scored.append((score + paper.get("relevance", 0), paper))
        top = [p for _, p in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]

        answer_parts = []
        citations = []
        for paper in top:
            snippet = paper.get("summary") or paper.get("abstract") or paper.get("title")
            answer_parts.append(f"{paper['title']}: {snippet}")
            citations.append({"pmid": paper["pmid"], "title": paper["title"], "url": paper["url"]})

        return {
            "answer": "\n\n".join(answer_parts),
            "citations": citations,
        }
