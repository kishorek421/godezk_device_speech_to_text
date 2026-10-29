"""Hybrid Semantic and Lexical Retrieval Engine.

Combines SQLite FTS5 BM25 keyword matching with cached dense text embeddings
generated via the remote Ollama embedding service (e.g. nomic-embed-text).
Rankings are fused using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, NamedTuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://124.123.89.162:11434")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
def resolve_default_db_path() -> str:
    env_path = os.getenv("HYBRID_RETRIEVAL_DB_PATH")
    if env_path:
        return env_path
    parents = Path(__file__).resolve().parents
    candidates = [
        Path("config/hybrid_retrieval.db"),
        parents[3] / "config" / "hybrid_retrieval.db" if len(parents) > 3 else None,
        parents[2] / "config" / "hybrid_retrieval.db" if len(parents) > 2 else None,
    ]
    for c in candidates:
        if c and c.exists():
            return str(c)
    return str(Path.cwd() / "config" / "hybrid_retrieval.db")


DEFAULT_DB_PATH = resolve_default_db_path()
DEFAULT_TIMEOUT_SEC = float(os.getenv("OLLAMA_EMBED_TIMEOUT_SEC", "5.0"))


class RetrievedPassage(NamedTuple):
    id: str
    text: str
    domain: str
    source: str
    category: str
    rrf_score: float
    lexical_rank: int | None
    semantic_rank: int | None
    lexical_score: float
    semantic_score: float
    metadata: dict[str, Any]


class HybridContextRetriever:
    """Combines SQLite FTS5 BM25 search with cached Ollama text embeddings."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        ollama_base_url: str = DEFAULT_OLLAMA_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ):
        self.db_path = Path(db_path)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.timeout_sec = timeout_sec
        self.embedding_dim: int | None = None

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_database(self) -> None:
        """Initializes SQLite tables and FTS5 index."""
        with self._get_connection() as conn:
            # Metadata and content table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    approval_status TEXT NOT NULL DEFAULT 'approved',
                    content_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            # Index on domain and approval_status
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_domain_status ON documents(domain, approval_status);"
            )

            # Embeddings storage
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(id) REFERENCES documents(id) ON DELETE CASCADE
                );
                """
            )

            # Check FTS5 availability and create virtual table
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                        id UNINDEXED,
                        text,
                        domain,
                        content='documents',
                        content_rowid='rowid'
                    );
                    """
                )
                # Triggers to keep FTS5 synchronized with documents table
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS trg_docs_ai AFTER INSERT ON documents BEGIN
                        INSERT INTO documents_fts(rowid, id, text, domain)
                        VALUES (new.rowid, new.id, new.text, new.domain);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS trg_docs_ad AFTER DELETE ON documents BEGIN
                        INSERT INTO documents_fts(documents_fts, rowid, id, text, domain)
                        VALUES('delete', old.rowid, old.id, old.text, old.domain);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS trg_docs_au AFTER UPDATE ON documents BEGIN
                        INSERT INTO documents_fts(documents_fts, rowid, id, text, domain)
                        VALUES('delete', old.rowid, old.id, old.text, old.domain);
                        INSERT INTO documents_fts(rowid, id, text, domain)
                        VALUES (new.rowid, new.id, new.text, new.domain);
                    END;
                    """
                )
            except sqlite3.OperationalError as exc:
                logger.error("SQLite FTS5 initialization error: %s", exc)
                raise RuntimeError(f"SQLite FTS5 is not supported on this platform: {exc}")

    @staticmethod
    def compute_content_hash(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

    def generate_embedding(self, text: str) -> np.ndarray | None:
        """Fetches embedding vector from remote Ollama server.
        
        Returns float32 numpy array or None if service is unavailable.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        url = f"{self.ollama_base_url}/api/embed"
        payload = {"model": self.embedding_model, "input": clean_text}

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings")
                if embeddings and len(embeddings) > 0 and isinstance(embeddings[0], (list, tuple)) and len(embeddings[0]) > 0:
                    vec = np.array(embeddings[0], dtype=np.float32)
                    if self.embedding_dim is not None and len(vec) != self.embedding_dim:
                        logger.warning(
                            "Rejecting embedding with unexpected dimension %d (expected %d)",
                            len(vec),
                            self.embedding_dim,
                        )
                        return None
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    if self.embedding_dim is None:
                        self.embedding_dim = len(vec)
                    return vec
            logger.warning(
                "Ollama embed call failed with HTTP %d: %s", resp.status_code, resp.text[:200]
            )
        except requests.Timeout:
            logger.warning("Ollama embedding generation timed out (>%.1fs)", self.timeout_sec)
        except Exception as exc:
            logger.warning("Failed to reach Ollama embedding endpoint: %s", exc)

        return None

    def add_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        force_reembed: bool = False,
    ) -> str:
        """Adds or updates an approved knowledge passage with automatic embedding caching.
        
        Idempotent: Reuses existing embeddings if content hash is unchanged.
        """
        meta = dict(metadata or {})
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Document text cannot be empty")

        doc_id = str(meta.get("id") or meta.get("document_id") or f"doc_{self.compute_content_hash(clean_text)}")
        domain = str(meta.get("domain", "general")).strip().lower()
        source = str(meta.get("source", "manual")).strip()
        category = str(meta.get("category", "terminology")).strip()
        approval_status = str(meta.get("approval_status", "approved")).strip()
        content_hash = self.compute_content_hash(clean_text)
        now = time.time()

        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT id, content_hash, version FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()

            needs_embedding = force_reembed or (existing is None) or (existing["content_hash"] != content_hash)

            if existing:
                new_version = existing["version"] + 1
                conn.execute(
                    """
                    UPDATE documents
                    SET text = ?, domain = ?, source = ?, category = ?, approval_status = ?,
                        content_hash = ?, updated_at = ?, version = ?, metadata_json = ?
                    WHERE id = ?;
                    """,
                    (
                        clean_text,
                        domain,
                        source,
                        category,
                        approval_status,
                        content_hash,
                        now,
                        new_version,
                        json.dumps(meta),
                        doc_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO documents (
                        id, text, domain, source, category, approval_status,
                        content_hash, created_at, updated_at, version, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        doc_id,
                        clean_text,
                        domain,
                        source,
                        category,
                        approval_status,
                        content_hash,
                        now,
                        now,
                        1,
                        json.dumps(meta),
                    ),
                )

            # Update cached vector embedding if required and document is approved
            if needs_embedding and approval_status == "approved":
                vec = self.generate_embedding(clean_text)
                if vec is not None:
                    blob = vec.tobytes()
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO document_embeddings (id, embedding, dim, model_name, updated_at)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (doc_id, blob, len(vec), self.embedding_model, now),
                    )

        return doc_id

    def build_or_update_index(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 16,
    ) -> int:
        """Batch indexes multiple passages with cached embeddings."""
        count = 0
        for doc in documents:
            text = doc.get("text", "")
            if text:
                self.add_document(text, metadata=doc)
                count += 1
        return count

    def remove_document(self, document_id: str) -> bool:
        """Deletes a document and its cached embedding."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.execute("DELETE FROM document_embeddings WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def get_index_stats(self) -> dict[str, Any]:
        """Returns statistics on active documents and cached embeddings."""
        with self._get_connection() as conn:
            total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            approved_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE approval_status = 'approved'"
            ).fetchone()[0]
            pending_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE approval_status = 'pending'"
            ).fetchone()[0]
            total_embeddings = conn.execute(
                "SELECT COUNT(*) FROM document_embeddings"
            ).fetchone()[0]
            domains = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT domain FROM documents ORDER BY domain"
                ).fetchall()
            ]

        return {
            "total_documents": total_docs,
            "approved_documents": approved_docs,
            "pending_documents": pending_docs,
            "total_embeddings": total_embeddings,
            "domains": domains,
            "embedding_model": self.embedding_model,
            "db_path": str(self.db_path),
        }

    # -----------------------------------------------------------------------
    # Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
    # -----------------------------------------------------------------------

    def search_lexical(
        self,
        query: str,
        domain: str = "general",
        limit: int = 10,
        cross_domain: bool = False,
    ) -> list[tuple[str, float]]:
        """Performs SQLite FTS5 BM25 search.
        
        Returns list of (document_id, bm25_score) in ascending order of score
        (in SQLite bm25(), lower/more negative is better match).
        """
        import re

        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        tokens = [t for t in clean_q.split() if len(t) >= 2]
        if not tokens:
            return []

        # SQLite FTS5 MATCH query with OR chaining between terms
        fts_expr = " OR ".join(f'"{t}"*' for t in tokens)

        dom = (domain or "general").strip().lower()
        sql = """
            SELECT d.id, bm25(documents_fts) AS rank_score
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.id
            WHERE documents_fts MATCH ?
              AND d.approval_status = 'approved'
        """
        params: list[Any] = [fts_expr]
        if not cross_domain and dom != "all":
            sql += " AND (d.domain = ? OR d.domain = 'general')"
            params.append(dom)

        sql += " ORDER BY rank_score ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
                # Return (id, -score) so higher is better
                return [(row["id"], -float(row["rank_score"])) for row in rows]
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 query failed for expr '%s': %s", fts_expr, exc)
                return []

    def search_semantic(
        self,
        query_vector: np.ndarray,
        domain: str = "general",
        limit: int = 10,
        cross_domain: bool = False,
    ) -> list[tuple[str, float]]:
        """Performs cosine similarity search against cached embeddings.
        
        Returns list of (document_id, cosine_similarity) sorted descending.
        """
        dom = (domain or "general").strip().lower()
        sql = """
            SELECT e.id, e.embedding, e.dim
            FROM document_embeddings e
            JOIN documents d ON d.id = e.id
            WHERE d.approval_status = 'approved'
        """
        params: list[Any] = []
        if not cross_domain and dom != "all":
            sql += " AND (d.domain = ? OR d.domain = 'general')"
            params.append(dom)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return []

            results = []
            for row in rows:
                doc_id = row["id"]
                blob = row["embedding"]
                dim = row["dim"]
                if len(blob) != dim * 4:  # float32 = 4 bytes
                    continue
                vec = np.frombuffer(blob, dtype=np.float32)
                if len(vec) != len(query_vector):
                    continue
                # Normalized dot product = cosine similarity
                cos_sim = float(np.dot(query_vector, vec))
                results.append((doc_id, cos_sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

    def search(
        self,
        query: str,
        domain: str = "general",
        top_k: int = 3,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
        cross_domain: bool = False,
    ) -> list[RetrievedPassage]:
        """Hybrid retrieval combining FTS5 lexical BM25 and Ollama dense cosine similarity.
        
        Fuses ranks using Reciprocal Rank Fusion (RRF):
            score(d) = (lexical_weight / (rrf_k + rank_lex(d))) + (semantic_weight / (rrf_k + rank_sem(d)))
        
        Safely falls back to lexical retrieval if the embedding service is offline.
        """
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        candidate_pool_limit = max(top_k * 4, 15)

        t_start = time.time()

        # 1. Lexical candidate ranking
        t_lex0 = time.time()
        lex_candidates = self.search_lexical(
            clean_q, domain=domain, limit=candidate_pool_limit, cross_domain=cross_domain
        )
        t_lex = time.time() - t_lex0

        # 2. Semantic candidate ranking
        t_emb0 = time.time()
        query_vec = self.generate_embedding(clean_q)
        t_emb = time.time() - t_emb0

        t_sem0 = time.time()
        sem_candidates: list[tuple[str, float]] = []
        if query_vec is not None:
            sem_candidates = self.search_semantic(
                query_vec, domain=domain, limit=candidate_pool_limit, cross_domain=cross_domain
            )
        t_sem = time.time() - t_sem0

        # Build rank maps
        lex_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(lex_candidates)}
        lex_scores = {doc_id: score for doc_id, score in lex_candidates}

        sem_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sem_candidates)}
        sem_scores = {doc_id: score for doc_id, score in sem_candidates}

        all_doc_ids = set(lex_ranks.keys()) | set(sem_ranks.keys())
        if not all_doc_ids:
            return []

        # 3. Reciprocal Rank Fusion
        fused_scores: dict[str, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in lex_ranks:
                score += lexical_weight / (rrf_k + lex_ranks[doc_id])
            if doc_id in sem_ranks:
                score += semantic_weight / (rrf_k + sem_ranks[doc_id])
            fused_scores[doc_id] = score

        # Sort by RRF score descending
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)[:top_k]

        # 4. Fetch passage metadata
        passages: list[RetrievedPassage] = []
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in sorted_doc_ids)
            sql = f"""
                SELECT id, text, domain, source, category, metadata_json
                FROM documents
                WHERE id IN ({placeholders});
            """
            rows = {r["id"]: r for r in conn.execute(sql, sorted_doc_ids).fetchall()}

            for doc_id in sorted_doc_ids:
                row = rows.get(doc_id)
                if not row:
                    continue
                meta = {}
                try:
                    meta = json.loads(row["metadata_json"])
                except Exception:
                    pass

                passages.append(
                    RetrievedPassage(
                        id=doc_id,
                        text=row["text"],
                        domain=row["domain"],
                        source=row["source"],
                        category=row["category"],
                        rrf_score=round(fused_scores[doc_id], 6),
                        lexical_rank=lex_ranks.get(doc_id),
                        semantic_rank=sem_ranks.get(doc_id),
                        lexical_score=round(lex_scores.get(doc_id, 0.0), 4),
                        semantic_score=round(sem_scores.get(doc_id, 0.0), 4),
                        metadata=meta,
                    )
                )

        total_retrieval_ms = round((time.time() - t_start) * 1000, 1)
        logger.debug(
            "Hybrid retrieval: %d results for domain '%s' (lex: %.1fms, emb: %.1fms, sem: %.1fms, total: %.1fms)",
            len(passages),
            domain,
            t_lex * 1000,
            t_emb * 1000,
            t_sem * 1000,
            total_retrieval_ms,
        )

        return passages

    # -----------------------------------------------------------------------
    # Bootstrap Knowledge Base
    # -----------------------------------------------------------------------

    def bootstrap_default_knowledge(self, force: bool = False) -> int:
        """Populates the database with initial approved knowledge from
        domain dictionaries and verified dataset manifests.
        """
        stats = self.get_index_stats()
        if stats["approved_documents"] > 0 and not force:
            logger.info("Hybrid retrieval index already populated (%d docs)", stats["approved_documents"])
            return stats["approved_documents"]

        docs_to_index: list[dict[str, Any]] = []

        # 1. Hospital Knowledge
        hospital_entries = [
            ("SpO2 represents peripheral capillary oxygen saturation. Normal range 95-100%.", "abbreviation", "SpO2"),
            ("BP represents arterial blood pressure measured in mmHg as systolic over diastolic.", "abbreviation", "BP"),
            ("ECG or EKG represents electrocardiogram monitoring cardiac electrical activity.", "abbreviation", "ECG"),
            ("Hypertension is high blood pressure with systolic >= 130 or diastolic >= 80 mmHg.", "terminology", "hypertension"),
            ("Tachycardia refers to a heart rate exceeding 100 beats per minute at rest.", "terminology", "tachycardia"),
            ("Arrhythmia is an abnormal heartbeat rhythm or irregular cardiac conduction.", "terminology", "arrhythmia"),
            ("Potassium penicillin is prescribed for bacterial infection post procedural pain management.", "medication", "penicillin"),
            ("Intravenous IV infusion delivers fluid, electrolytes or medication directly into the vein.", "procedure", "IV"),
            ("Pneumonia is an infection that inflames the alveoli in one or both lungs.", "terminology", "pneumonia"),
            ("PEEP represents positive end-expiratory pressure during mechanical ventilation.", "abbreviation", "PEEP"),
        ]
        for text, cat, term in hospital_entries:
            docs_to_index.append({
                "id": f"hosp_{term.lower()}",
                "text": text,
                "domain": "hospital",
                "source": "clinical_standards",
                "category": cat,
                "approval_status": "approved",
                "term": term,
            })

        # 2. Education Knowledge
        education_entries = [
            ("pH measures acidity or alkalinity on a logarithmic scale from 0 to 14 where 7 is neutral.", "abbreviation", "pH"),
            ("HCl represents hydrochloric acid, a strong monoprotic mineral acid.", "formula", "HCl"),
            ("NaOH represents sodium hydroxide, a strong caustic metallic base used in neutralization.", "formula", "NaOH"),
            ("H2SO4 represents sulfuric acid, a strong diprotic mineral acid and oxidizing agent.", "formula", "H2SO4"),
            ("Titration is quantitative chemical analysis determining the unknown concentration of an identified analyte.", "terminology", "titration"),
            ("Molarity represents molar concentration measured as moles of solute per liter of solution.", "terminology", "molarity"),
            ("Thermodynamics governs heat, work, entropy and energy conservation in physical systems.", "terminology", "thermodynamics"),
            ("Kinetic energy KE is energy possessed by an object due to its motion equal to half mass times velocity squared.", "abbreviation", "KE"),
            ("Potential energy PE is stored energy due to an object's relative position or configuration.", "abbreviation", "PE"),
            ("Quantum electrodynamics QED is the relativistic quantum field theory of electrodynamics.", "terminology", "quantum"),
            ("Photosynthesis is the cellular process by which plants use sunlight to synthesize nutrients.", "terminology", "photosynthesis"),
            ("Mitochondria are double-membraned cellular organelles generating adenosine triphosphate ATP.", "terminology", "mitochondria"),
        ]
        for text, cat, term in education_entries:
            docs_to_index.append({
                "id": f"edu_{term.lower()}",
                "text": text,
                "domain": "education",
                "source": "academic_curriculum",
                "category": cat,
                "approval_status": "approved",
                "term": term,
            })

        # 3. Industry Knowledge
        industry_entries = [
            ("LOTO represents lockout-tagout safety procedure ensuring hazardous energy isolation before servicing.", "abbreviation", "LOTO"),
            ("PPE represents personal protective equipment including hard hats, eye protection and steel-toe boots.", "abbreviation", "PPE"),
            ("PLC represents programmable logic controller used for industrial automation and machinery control.", "abbreviation", "PLC"),
            ("Emergency stop activated on the automated conveyor assembly line isolates motive power immediately.", "safety_sop", "emergency_stop"),
            ("Transformer primary winding and secondary winding step up or step down alternating current voltage.", "equipment", "transformer"),
            ("Circuit breaker automatically interrupts electrical current flow when an overload or short circuit occurs.", "equipment", "circuit_breaker"),
            ("Pneumatic and hydraulic actuators convert pressurized fluid into controlled mechanical motion.", "equipment", "actuator"),
            ("CNC computer numerical control automates machining tools via programmed sequences.", "abbreviation", "CNC"),
            ("HVAC represents heating, ventilation, and air conditioning maintaining environmental conditions.", "abbreviation", "HVAC"),
            ("Overall equipment effectiveness OEE measures manufacturing productivity efficiency.", "abbreviation", "OEE"),
        ]
        for text, cat, term in industry_entries:
            docs_to_index.append({
                "id": f"ind_{term.lower()}",
                "text": text,
                "domain": "industry",
                "source": "industrial_safety_sop",
                "category": cat,
                "approval_status": "approved",
                "term": term,
            })

        # 4. Verified Manifest Samples (ground-truth training examples)
        manifest_path = Path(__file__).resolve().parents[2] / "dataset" / "manifests" / "full_manifest.jsonl"
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        text = entry.get("text", "")
                        domain = entry.get("domain", "general")
                        if text and len(text) > 10:
                            docs_to_index.append({
                                "id": f"manifest_{domain}_{idx}",
                                "text": text,
                                "domain": domain,
                                "source": "verified_speech_manifest",
                                "category": "verified_transcript",
                                "approval_status": "approved",
                            })
            except Exception as exc:
                logger.warning("Could not ingest full_manifest.jsonl: %s", exc)

        logger.info("Bootstrapping %d approved knowledge documents...", len(docs_to_index))
        indexed = self.build_or_update_index(docs_to_index)
        logger.info("Successfully indexed %d approved knowledge documents", indexed)
        return indexed
