"""
Entity linker and query expander using the Knowledge Graph.

Given a user query, finds matching KG entities and expands the query
with canonical terms and related concepts for improved retrieval.
"""

import logging
import re

from agribot.knowledge_graph.schema import KnowledgeGraph, Entity

logger = logging.getLogger(__name__)


class EntityLinker:
    """Links query terms to KG entities and expands queries."""

    def __init__(self, kg: KnowledgeGraph, expansion_hops: int = 1):
        self.kg = kg
        self.expansion_hops = expansion_hops

    def _tokenize_query(self, query: str) -> list[str]:
        """Split query into candidate tokens and bigrams."""
        # Clean and split
        words = re.findall(r"[\w\u0980-\u09FF]+", query.lower())

        # Generate unigrams and bigrams
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]} {words[i + 1]}")

        # Also add trigrams for compound terms
        for i in range(len(words) - 2):
            tokens.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")

        return tokens

    def link_entities(self, query: str) -> list[Entity]:
        """
        Find KG entities that match terms in the query.

        Tries exact alias matching first, then partial matching.
        """
        tokens = self._tokenize_query(query)
        found_entities: dict[int, Entity] = {}

        # Exact alias match (highest priority)
        for token in tokens:
            entities = self.kg.find_by_alias(token)
            for entity in entities:
                if entity.id not in found_entities:
                    found_entities[entity.id] = entity

        # Partial alias match (lower priority, only if few exact matches)
        if len(found_entities) < 3:
            for token in tokens:
                if len(token) < 3:
                    continue
                entities = self.kg.find_by_partial_alias(token)
                for entity in entities:
                    if entity.id not in found_entities:
                        found_entities[entity.id] = entity

        logger.info(
            "Linked %d entities from query: %s",
            len(found_entities), list(found_entities.keys()),
        )
        return list(found_entities.values())

    def expand_query(self, query: str) -> str:
        """
        Expand a query with KG-linked terms for better retrieval.

        Process:
        1. Find matching entities in query
        2. Get canonical BN/EN terms
        3. Get related entities through graph expansion
        4. Combine into expanded query

        Args:
            query: Original user query

        Returns:
            Expanded query string with canonical and related terms
        """
        linked = self.link_entities(query)
        if not linked:
            logger.info("No entities linked, returning original query")
            return query

        expansion_terms: set[str] = set()

        for entity in linked:
            # Add canonical terms
            if entity.canonical_bn:
                expansion_terms.add(entity.canonical_bn)
            if entity.canonical_en:
                expansion_terms.add(entity.canonical_en)

            # Get aliases (other dialect forms)
            aliases = self.kg.get_aliases(entity.id)
            for alias in aliases:
                expansion_terms.add(alias.alias_text)

            # Graph expansion: get related entities
            neighbors = self.kg.get_neighbors(
                entity.id, hops=self.expansion_hops
            )
            for neighbor_entity, rel_type in neighbors:
                expansion_terms.add(neighbor_entity.canonical_en)
                if neighbor_entity.canonical_bn:
                    expansion_terms.add(neighbor_entity.canonical_bn)

        # Build expanded query: original + expansion terms
        # Remove terms already in the original query
        query_lower = query.lower()
        new_terms = [
            t for t in expansion_terms
            if t.lower() not in query_lower and len(t) > 1
        ]

        if new_terms:
            expanded = f"{query} {' '.join(new_terms)}"
            logger.info("Expanded query with %d terms: %s", len(new_terms), new_terms)
            return expanded

        return query
