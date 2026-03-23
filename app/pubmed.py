from __future__ import annotations

import os
import time
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "research_navigator_ai"
EMAIL = os.getenv("NCBI_EMAIL", "example@example.com")
API_KEY = os.getenv("NCBI_API_KEY")


class PubMedClient:
    def __init__(self) -> None:
        self.session = requests.Session()

    def _common_params(self) -> Dict[str, str]:
        params = {"tool": TOOL_NAME, "email": EMAIL}
        if API_KEY:
            params["api_key"] = API_KEY
        return params

    def _get(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        merged = {**self._common_params(), **params}
        resp = self.session.get(f"{BASE_URL}/{endpoint}", params=merged, timeout=60)
        resp.raise_for_status()
        # be polite to NCBI; stay well under rate limits
        time.sleep(0.34 if not API_KEY else 0.12)
        return resp

    def search_pmids(self, query: str, max_results: int = 20) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        }
        data = self._get("esearch.fcgi", params).json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        xml_text = self._get("efetch.fcgi", params).text
        root = ET.fromstring(xml_text)
        articles: List[Dict[str, Any]] = []

        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue

            pmid = self._safe_text(medline.find("PMID"))
            art = medline.find("Article")
            if art is None:
                continue

            title = self._collect_text(art.find("ArticleTitle"))
            abstract_parts = []
            for node in art.findall(".//Abstract/AbstractText"):
                label = node.attrib.get("Label")
                text = self._collect_text(node)
                if text:
                    abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = "\n".join(abstract_parts).strip()

            journal = self._safe_text(art.find(".//Journal/Title"))
            year = self._extract_year(art)
            authors = []
            for a in art.findall(".//AuthorList/Author")[:8]:
                last = self._safe_text(a.find("LastName"))
                fore = self._safe_text(a.find("ForeName"))
                collective = self._safe_text(a.find("CollectiveName"))
                if collective:
                    authors.append(collective)
                elif last:
                    authors.append(f"{fore} {last}".strip())

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": authors,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        return articles

    @staticmethod
    def _safe_text(node: ET.Element | None) -> str:
        return "" if node is None or node.text is None else node.text.strip()

    @staticmethod
    def _collect_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return "".join(node.itertext()).strip()

    @staticmethod
    def _extract_year(article_node: ET.Element) -> str:
        year = article_node.find(".//JournalIssue/PubDate/Year")
        medline_date = article_node.find(".//JournalIssue/PubDate/MedlineDate")
        if year is not None and year.text:
            return year.text.strip()
        if medline_date is not None and medline_date.text:
            return medline_date.text.strip()[:4]
        return "Unknown"
