"""
Answer Generation — Phase 1 baseline.

Formats retrieved triples into an IC-HRAG-style prompt template and
calls an LLM for answer generation. Produces structured JSON output
with answer + reasoning trace.

Phase 1: Uses a local model via transformers (or falls back to a
simple extractive baseline for testing without GPU).
Phase 4+: Will be extended with structured in-model decryption hooks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from shield_rag.schema.ontology import RetrievedTriple


# ─── Prompt Templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise technical assistant. You answer questions about industrial equipment based ONLY on the provided knowledge graph triples. Each triple has the form (Entity1 –Relation→ Entity2).

Rules:
1. Only use information from the provided triples. Do not use external knowledge.
2. If the triples do not contain enough information to answer, say "INSUFFICIENT_EVIDENCE".
3. For true/false questions, evaluate the condition against the evidence and give a clear TRUE or FALSE verdict.
4. Always cite which triples support your answer.
5. Respond in the exact JSON format specified."""

USER_PROMPT_TEMPLATE = """## Knowledge Graph Context
{triples_text}

## Query
{query}

## Required Output Format
Respond with a JSON object containing:
{{
    "query_intent": "<brief description of what the query asks>",
    "evidence_triples": [<indices of triples used, 0-indexed>],
    "reasoning": "<step-by-step reasoning using only the triples>",
    "answer": "<the answer>",
    "verdict": "<TRUE|FALSE|INSUFFICIENT_EVIDENCE for true/false questions, null otherwise>",
    "confidence": <float 0-1>
}}"""


@dataclass
class GenerationConfig:
    """Configuration for the answer generator."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    use_local_model: bool = True
    fallback_to_extractive: bool = True  # use extractive baseline if no GPU


@dataclass
class AnswerResult:
    """Structured output from the answer generator."""
    query: str = ""
    query_intent: str = ""
    evidence_triples: list[int] = field(default_factory=list)
    reasoning: str = ""
    answer: str = ""
    verdict: Optional[str] = None  # TRUE, FALSE, INSUFFICIENT_EVIDENCE, or None
    confidence: float = 0.0
    raw_response: str = ""
    triples_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_intent": self.query_intent,
            "evidence_triples": self.evidence_triples,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "triples_used": self.triples_used,
            "latency_ms": self.latency_ms,
        }


def format_triples_for_prompt(triples: list[RetrievedTriple]) -> str:
    """Format retrieved triples into numbered text for the prompt.

    Uses IC-HRAG's knowledge-integration format:
    [i] (Entity1 –Relation→ Entity2) [score: X.XX]
    """
    lines = []
    for i, triple in enumerate(triples):
        lines.append(f"[{i}] {triple.to_text()} [score: {triple.score:.3f}]")
    return "\n".join(lines)


class AnswerGenerator:
    """Generates answers from retrieved triples using an LLM.

    Phase 1 implementation supports:
    1. Local model via transformers (requires GPU)
    2. Extractive baseline (no GPU needed) — pattern-matches triples
       against the query for true/false questions

    The output interface (AnswerResult) is frozen for Phase 4+ compatibility.
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        self._config = config or GenerationConfig()
        self._model = None
        self._tokenizer = None
        self._use_extractive = False

    def _load_model(self) -> None:
        """Lazy-load the LLM. Falls back to extractive if unavailable."""
        if self._model is not None or self._use_extractive:
            return

        if not self._config.use_local_model:
            self._use_extractive = True
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                self._model = self._model.to(device)
        except Exception:
            if self._config.fallback_to_extractive:
                self._use_extractive = True
            else:
                raise

    def generate(
        self,
        query: str,
        triples: list[RetrievedTriple],
    ) -> AnswerResult:
        """Generate an answer from retrieved triples.

        Args:
            query:   The user's question.
            triples: Retrieved (head, relation, tail) triples with scores.

        Returns:
            AnswerResult with structured answer, reasoning, and verdict.
        """
        import time

        start = time.perf_counter()

        self._load_model()

        triples_text = format_triples_for_prompt(triples)
        triple_strings = [t.to_text() for t in triples]

        if self._use_extractive:
            result = self._extractive_answer(query, triples, triples_text)
        else:
            result = self._llm_answer(query, triples_text)

        result.query = query
        result.triples_used = triple_strings
        result.latency_ms = (time.perf_counter() - start) * 1000

        return result

    def _llm_answer(self, query: str, triples_text: str) -> AnswerResult:
        """Generate answer using the local LLM."""
        import torch

        user_prompt = USER_PROMPT_TEMPLATE.format(
            triples_text=triples_text,
            query=query,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(
            self._model.device
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                do_sample=self._config.temperature > 0,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return self._parse_response(response)

    def _extractive_answer(
        self,
        query: str,
        triples: list[RetrievedTriple],
        triples_text: str,
    ) -> AnswerResult:
        """Simple extractive baseline for true/false questions.

        Pattern-matches query keywords against triple texts to determine
        answer. Good enough for Phase 1 baseline evaluation.
        """
        query_lower = query.lower()

        # Detect if it's a true/false question
        is_tf = any(
            kw in query_lower
            for kw in [
                "is the", "does the", "are the", "can the",
                "is it", "does it", "true or false",
                "above", "below", "exceed", "greater", "less",
            ]
        )

        # Find most relevant triples
        relevant_indices = []
        for i, triple in enumerate(triples):
            triple_text = triple.to_text().lower()
            # Check keyword overlap
            query_words = set(re.findall(r'\w+', query_lower))
            triple_words = set(re.findall(r'\w+', triple_text))
            overlap = len(query_words & triple_words)
            if overlap >= 2 or triple.score > 0.5:
                relevant_indices.append(i)

        if not relevant_indices:
            relevant_indices = list(range(min(3, len(triples))))

        # For true/false: extract numbers and compare
        verdict = None
        confidence = 0.5
        reasoning_parts = []

        if is_tf and triples:
            verdict, confidence, reasoning_parts = self._evaluate_condition(
                query_lower, triples, relevant_indices
            )

        # Build answer
        if verdict is not None:
            answer = f"{'TRUE' if verdict == 'TRUE' else 'FALSE'}: Based on the provided evidence."
        elif triples:
            # Extractive: return the most relevant triple's content
            best = triples[0]
            answer = f"Based on the knowledge graph: {best.to_text()}"
        else:
            answer = "INSUFFICIENT_EVIDENCE"
            verdict = "INSUFFICIENT_EVIDENCE"

        reasoning = " → ".join(reasoning_parts) if reasoning_parts else "Direct extraction from triples."

        return AnswerResult(
            query_intent="extractive_baseline",
            evidence_triples=relevant_indices,
            reasoning=reasoning,
            answer=answer,
            verdict=verdict,
            confidence=confidence,
            raw_response="[extractive baseline]",
        )

    @staticmethod
    def _evaluate_condition(
        query_lower: str,
        triples: list[RetrievedTriple],
        relevant_indices: list[int],
    ) -> tuple[Optional[str], float, list[str]]:
        """Evaluate a numeric condition from the query against triple evidence."""
        reasoning = []

        # Extract numbers from query
        query_numbers = re.findall(r'[\d,]+\.?\d*', query_lower)
        query_numbers_parsed = []
        for n in query_numbers:
            n_clean = n.replace(',', '').strip()
            if n_clean:
                try:
                    query_numbers_parsed.append(float(n_clean))
                except ValueError:
                    continue

        # Extract numbers from relevant triples
        triple_numbers = []
        for idx in relevant_indices:
            if idx < len(triples):
                text = triples[idx].to_text().lower()
                nums = re.findall(r'[\d,]+\.?\d*', text)
                for n in nums:
                    n_clean = n.replace(',', '').strip()
                    if n_clean:
                        try:
                            triple_numbers.append((float(n_clean), idx))
                        except ValueError:
                            continue

        if not query_numbers_parsed or not triple_numbers:
            # Can't do numeric comparison; use similarity-based heuristic
            if triples and triples[0].score > 0.6:
                reasoning.append("High similarity match found")
                return "TRUE", 0.6, reasoning
            return None, 0.3, reasoning

        query_val = query_numbers_parsed[0]
        evidence_val, evidence_idx = triple_numbers[0]

        reasoning.append(f"Query asks about value: {query_val}")
        reasoning.append(f"Evidence shows value: {evidence_val} (from triple [{evidence_idx}])")

        # Determine comparison direction from keywords
        if any(kw in query_lower for kw in ["above", "greater", "exceed", "over", "more"]):
            result = evidence_val > query_val
            reasoning.append(f"{evidence_val} > {query_val} = {result}")
            return "TRUE" if result else "FALSE", 0.85, reasoning
        elif any(kw in query_lower for kw in ["below", "less", "under", "fewer"]):
            result = evidence_val < query_val
            reasoning.append(f"{evidence_val} < {query_val} = {result}")
            return "TRUE" if result else "FALSE", 0.85, reasoning
        else:
            # Default: check if values are close
            result = abs(evidence_val - query_val) / max(abs(query_val), 1) < 0.1
            reasoning.append(f"Values match: {evidence_val} ≈ {query_val} = {result}")
            return "TRUE" if result else "FALSE", 0.7, reasoning

    @staticmethod
    def _parse_response(response: str) -> AnswerResult:
        """Parse LLM JSON response into AnswerResult."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return AnswerResult(
                    query_intent=data.get("query_intent", ""),
                    evidence_triples=data.get("evidence_triples", []),
                    reasoning=data.get("reasoning", ""),
                    answer=data.get("answer", ""),
                    verdict=data.get("verdict"),
                    confidence=float(data.get("confidence", 0.5)),
                    raw_response=response,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: treat entire response as the answer
        return AnswerResult(
            query_intent="unparsed",
            reasoning="Could not parse structured response.",
            answer=response.strip(),
            confidence=0.3,
            raw_response=response,
        )
