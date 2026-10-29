"""Daily Learning and Human Approval Workflow Manager.

Collects human-verified transcription feedback, identifies recurring domain errors,
generates pending improvement suggestions, enforces human approval before updating
dictionaries/retrieval indices, and supports rolling back approved snapshots.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any

logger = logging.getLogger(__name__)

def resolve_default_config_dir() -> Path:
    env_dir = os.getenv("HYBRID_RETRIEVAL_CONFIG_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    parents = Path(__file__).resolve().parents
    candidates = [
        Path.cwd() / "config",
        parents[3] / "config" if len(parents) > 3 else None,
        parents[2] / "config" if len(parents) > 2 else None,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return Path.cwd() / "config"


DEFAULT_CONFIG_DIR = resolve_default_config_dir()
DEFAULT_SUGGESTIONS_FILE = DEFAULT_CONFIG_DIR / "learning_suggestions.json"
DEFAULT_FEEDBACK_FILE = DEFAULT_CONFIG_DIR / "verified_feedback_log.jsonl"
DEFAULT_DICT_FILE = DEFAULT_CONFIG_DIR / "domain_dictionary.json"
DEFAULT_SNAPSHOTS_DIR = DEFAULT_CONFIG_DIR / "snapshots"


class DailyLearningManager:
    """Manages verified speech feedback, suggestion generation, human approval, and rollback."""

    def __init__(
        self,
        config_dir: str | Path = DEFAULT_CONFIG_DIR,
        suggestions_path: str | Path | None = None,
        feedback_path: str | Path | None = None,
        dict_path: str | Path | None = None,
        snapshots_dir: str | Path | None = None,
    ):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.suggestions_path = Path(suggestions_path) if suggestions_path else self.config_dir / "learning_suggestions.json"
        self.feedback_path = Path(feedback_path) if feedback_path else self.config_dir / "verified_feedback_log.jsonl"
        if dict_path:
            self.dict_path = Path(dict_path)
        else:
            try:
                from walker_audio.dictionary_corrector import find_dictionary_path
                found = find_dictionary_path()
                self.dict_path = Path(found) if found else self.config_dir / "domain_dictionary.json"
            except Exception:
                self.dict_path = self.config_dir / "domain_dictionary.json"

        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else self.config_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_files()

    def _ensure_files(self) -> None:
        if not self.suggestions_path.exists():
            self._save_suggestions([])
        if not self.feedback_path.exists():
            self.feedback_path.touch()

    def _load_suggestions(self) -> list[dict[str, Any]]:
        if not self.suggestions_path.exists():
            return []
        try:
            with open(self.suggestions_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not read suggestions file %s: %s", self.suggestions_path, exc)
            return []

    def _save_suggestions(self, suggestions: list[dict[str, Any]]) -> None:
        tmp_file = self.suggestions_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(suggestions, f, indent=2, ensure_ascii=False)
        tmp_file.replace(self.suggestions_path)

    # -----------------------------------------------------------------------
    # 1. Collect Verified Feedback
    # -----------------------------------------------------------------------

    def record_verified_feedback(
        self,
        original_transcript: str,
        verified_transcript: str,
        domain: str = "general",
        source_sample_id: str | None = None,
        speaker_id: str | None = None,
        reviewer: str = "human_curator",
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Appends human-confirmed ground truth to the feedback log (idempotently)."""
        orig_clean = original_transcript.strip()
        verif_clean = verified_transcript.strip()
        dom = domain.strip().lower()

        if not verif_clean:
            raise ValueError("Verified transcript cannot be empty")

        entry_hash = hashlib.sha256(
            f"{orig_clean.lower()}|{verif_clean.lower()}|{dom}".encode("utf-8")
        ).hexdigest()[:16]

        entry = {
            "id": f"fb_{entry_hash}",
            "original_transcript": orig_clean,
            "verified_transcript": verif_clean,
            "domain": dom,
            "source_sample_id": source_sample_id,
            "speaker_id": speaker_id,
            "reviewer": reviewer,
            "notes": notes,
            "recorded_at": time.time(),
        }

        # Check for duplicate in log to keep idempotent
        if self.feedback_path.exists():
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if item.get("id") == entry["id"]:
                                return item  # Already recorded
                        except Exception:
                            pass

        with open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("Recorded verified feedback [%s]: '%s' -> '%s'", dom, orig_clean, verif_clean)
        return entry

    # -----------------------------------------------------------------------
    # 2. Analyze Recurring Mistakes and Generate Suggestions
    # -----------------------------------------------------------------------

    def generate_candidate_suggestions(
        self,
        min_frequency: int = 1,
        use_llm: bool = False,
    ) -> list[dict[str, Any]]:
        """Analyzes feedback logs and verified manifests to produce structured candidate suggestions.
        
        Every generated suggestion is created with status='pending'.
        """
        existing_suggestions = self._load_suggestions()
        existing_keys = {
            f"{s['domain']}|{s['original_phrase'].lower()}|{s['proposed_term'].lower()}"
            for s in existing_suggestions
        }

        # 1. Collect all verified pairs from feedback log
        feedback_pairs: list[dict[str, Any]] = []
        if self.feedback_path.exists():
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            feedback_pairs.append(json.loads(line))
                        except Exception:
                            pass

        # 2. Extract word-level divergence patterns
        # Grouped by (domain, original_token, corrected_token)
        divergence_counts: dict[tuple[str, str, str], list[str]] = {}

        for fb in feedback_pairs:
            dom = fb.get("domain", "general").lower()
            orig = fb.get("original_transcript", "")
            verif = fb.get("verified_transcript", "")
            if not orig or not verif or orig == verif:
                continue

            matcher = difflib.SequenceMatcher(None, orig.split(), verif.split())
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ("replace", "delete", "insert"):
                    sub_orig = " ".join(orig.split()[i1:i2]).strip()
                    sub_verif = " ".join(verif.split()[j1:j2]).strip()
                    if sub_orig and sub_verif and sub_orig.lower() != sub_verif.lower():
                        key = (dom, sub_orig, sub_verif)
                        if key not in divergence_counts:
                            divergence_counts[key] = []
                        divergence_counts[key].append(fb.get("source_sample_id") or fb.get("id"))

        # 3. Formulate pending suggestions for candidates meeting min_frequency
        new_suggestions: list[dict[str, Any]] = []
        for (dom, orig_phrase, prop_term), sample_ids in divergence_counts.items():
            if len(sample_ids) < min_frequency:
                continue

            key_str = f"{dom}|{orig_phrase.lower()}|{prop_term.lower()}"
            if key_str in existing_keys:
                continue

            # Determine suggestion type
            sug_type = "alias"
            if len(prop_term) <= 5 and prop_term.isupper():
                sug_type = "abbreviation"
            elif any(c.isupper() for c in prop_term[1:]):  # Chemical formula like HCl or pH
                sug_type = "formula"

            sug_id = f"sug_{hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:12]}"
            suggestion = {
                "id": sug_id,
                "domain": dom,
                "original_phrase": orig_phrase,
                "proposed_term": prop_term,
                "suggestion_type": sug_type,
                "reason": f"Acoustic ASR frequently misrecognized '{orig_phrase}' for verified term '{prop_term}' in domain '{dom}'.",
                "frequency": len(sample_ids),
                "evidence_sample_ids": sample_ids[:10],
                "status": "pending",
                "created_at": time.time(),
                "snapshot_file": None,
            }
            new_suggestions.append(suggestion)
            existing_suggestions.append(suggestion)
            existing_keys.add(key_str)

        if new_suggestions:
            self._save_suggestions(existing_suggestions)
            logger.info("Generated %d new pending suggestions", len(new_suggestions))
        else:
            logger.info("No new suggestions generated (insufficient recurring errors or already staged)")

        return new_suggestions

    # -----------------------------------------------------------------------
    # 3. View, Approve, Reject & Rollback
    # -----------------------------------------------------------------------

    def get_suggestions(self, status: str | None = None) -> list[dict[str, Any]]:
        """Returns suggestions optionally filtered by status ('pending', 'approved', 'rejected')."""
        all_sugs = self._load_suggestions()
        if not status or status.lower() == "all":
            return all_sugs
        st = status.lower()
        return [s for s in all_sugs if s.get("status") == st]

    def approve_suggestion(
        self,
        suggestion_id: str,
        reviewer: str = "human_curator",
        retriever: Any | None = None,
    ) -> dict[str, Any]:
        """Approves a pending suggestion, creates a rollback snapshot, updates domain_dictionary.json,
        and indexes the approved terminology into the hybrid retriever.
        """
        suggestions = self._load_suggestions()
        target: dict[str, Any] | None = None
        for s in suggestions:
            if s.get("id") == suggestion_id:
                target = s
                break

        if not target:
            raise KeyError(f"Suggestion ID '{suggestion_id}' not found")
        if target.get("status") == "approved":
            return target  # Idempotent

        # 1. Create a snapshot of domain_dictionary.json for rollback
        timestamp_str = int(time.time() * 1000)
        snapshot_name = f"dict_snapshot_{suggestion_id}_{timestamp_str}.json"
        snapshot_path = self.snapshots_dir / snapshot_name
        if self.dict_path.exists():
            shutil.copy2(self.dict_path, snapshot_path)

        # 2. Update domain_dictionary.json
        domain = target["domain"]
        orig = target["original_phrase"]
        prop = target["proposed_term"]

        dict_data: dict[str, Any] = {}
        if self.dict_path.exists():
            try:
                with open(self.dict_path, "r", encoding="utf-8") as f:
                    dict_data = json.load(f)
            except Exception:
                dict_data = {}

        # Support both flat domain dictionary format (e.g. {"medical": {"orig": "prop"}})
        # and nested format (e.g. {"domains": {"hospital": {"aliases": {...}}}})
        domain_key = domain.lower()
        if "domains" in dict_data:
            if domain_key in dict_data["domains"]:
                dict_domain = domain_key
            elif domain_key in ("hospital", "healthcare", "med"):
                dict_domain = "medical" if "medical" in dict_data["domains"] else domain_key
            else:
                dict_domain = domain_key

            if dict_domain not in dict_data["domains"]:
                dict_data["domains"][dict_domain] = {"terms": {}, "aliases": {}}
            elif "aliases" not in dict_data["domains"][dict_domain]:
                dict_data["domains"][dict_domain]["aliases"] = {}
            dict_data["domains"][dict_domain]["aliases"][orig] = prop
        else:
            if domain_key in dict_data:
                dict_domain = domain_key
            elif domain_key in ("hospital", "healthcare", "med"):
                dict_domain = "medical" if "medical" in dict_data else domain_key
            else:
                dict_domain = domain_key

            if dict_domain not in dict_data:
                dict_data[dict_domain] = {}
            dict_data[dict_domain][orig.lower()] = prop

        # Write updated dictionary atomically
        tmp_dict = self.dict_path.with_suffix(".tmp")
        with open(tmp_dict, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, indent=2, ensure_ascii=False)
        tmp_dict.replace(self.dict_path)

        # 3. Add to Retriever as approved document
        if retriever is not None:
            doc_text = f"Approved {domain} terminology: '{prop}' (replaces '{orig}')."
            retriever.add_document(
                text=doc_text,
                metadata={
                    "id": f"approved_{suggestion_id}",
                    "domain": domain,
                    "source": "daily_learning_approval",
                    "category": target.get("suggestion_type", "alias"),
                    "approval_status": "approved",
                    "original_phrase": orig,
                    "approved_term": prop,
                },
            )

        # 4. Mark suggestion as approved
        target["status"] = "approved"
        target["approved_at"] = time.time()
        target["approved_by"] = reviewer
        target["snapshot_file"] = str(snapshot_path.relative_to(self.config_dir))

        self._save_suggestions(suggestions)
        logger.info("Approved suggestion '%s': '%s' -> '%s' (domain: %s)", suggestion_id, orig, prop, domain)
        return target

    def reject_suggestion(
        self,
        suggestion_id: str,
        reviewer: str = "human_curator",
        reason: str = "",
    ) -> dict[str, Any]:
        """Marks a suggestion as rejected with supporting reasoning."""
        suggestions = self._load_suggestions()
        target: dict[str, Any] | None = None
        for s in suggestions:
            if s.get("id") == suggestion_id:
                target = s
                break

        if not target:
            raise KeyError(f"Suggestion ID '{suggestion_id}' not found")

        target["status"] = "rejected"
        target["rejected_at"] = time.time()
        target["rejected_by"] = reviewer
        target["rejection_reason"] = reason

        self._save_suggestions(suggestions)
        logger.info("Rejected suggestion '%s' by '%s': %s", suggestion_id, reviewer, reason)
        return target

    def rollback_suggestion(
        self,
        suggestion_id: str,
        retriever: Any | None = None,
    ) -> dict[str, Any]:
        """Rolls back an approved suggestion, restoring previous dictionary and removing index document."""
        suggestions = self._load_suggestions()
        target: dict[str, Any] | None = None
        for s in suggestions:
            if s.get("id") == suggestion_id:
                target = s
                break

        if not target:
            raise KeyError(f"Suggestion ID '{suggestion_id}' not found")
        if target.get("status") != "approved":
            raise ValueError(f"Suggestion '{suggestion_id}' is not in approved state (status: {target.get('status')})")

        # 1. Restore domain dictionary from snapshot if available
        snapshot_rel = target.get("snapshot_file")
        if snapshot_rel:
            snap_path = self.config_dir / snapshot_rel
            if snap_path.exists():
                shutil.copy2(snap_path, self.dict_path)
                logger.info("Restored domain dictionary from snapshot: %s", snap_path)
            else:
                # Remove alias manually from dictionary
                self._remove_alias_from_dict(target["domain"], target["original_phrase"])
        else:
            self._remove_alias_from_dict(target["domain"], target["original_phrase"])

        # 2. Remove document from retriever index
        if retriever is not None:
            retriever.remove_document(f"approved_{suggestion_id}")

        # 3. Reset suggestion status
        target["status"] = "pending"
        target["rolled_back_at"] = time.time()
        self._save_suggestions(suggestions)

        logger.info("Successfully rolled back suggestion '%s'", suggestion_id)
        return target

    def _remove_alias_from_dict(self, domain: str, original_phrase: str) -> None:
        if not self.dict_path.exists():
            return
        try:
            with open(self.dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            domain_key = domain.lower()
            candidates = [domain_key]
            if domain_key in ("hospital", "healthcare", "med"):
                candidates.append("medical")

            modified = False
            for d_key in candidates:
                if "domains" in data and d_key in data["domains"] and original_phrase in data["domains"][d_key].get("aliases", {}):
                    del data["domains"][d_key]["aliases"][original_phrase]
                    modified = True
                if d_key in data and original_phrase.lower() in data[d_key]:
                    del data[d_key][original_phrase.lower()]
                    modified = True

            if modified:
                with open(self.dict_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Could not manually remove alias '%s' from dictionary: %s", original_phrase, exc)
