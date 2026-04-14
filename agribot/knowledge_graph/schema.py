"""
SQLite-based Dialect Knowledge Graph.

Schema:
- entities: canonical agricultural terms (Bengali + English + entity type)
- aliases: dialect/colloquial term mappings to canonical entities
- relations: edges connecting entities (symptom→disease, disease→treatment, etc.)
"""

import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    id: int
    canonical_bn: str
    canonical_en: str
    entity_type: str  # crop, disease, pest, fertilizer, symptom, treatment, chemical


@dataclass
class Alias:
    id: int
    entity_id: int
    alias_text: str
    dialect_region: str  # e.g., "standard", "sylheti", "chittagong", "barishal"


@dataclass
class Relation:
    src_id: int
    rel_type: str  # symptom_of, treatment_for, causes, applied_to, etc.
    dst_id: int
    provenance: str


class KnowledgeGraph:
    """SQLite-backed dialect knowledge graph for agricultural term alignment."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_bn TEXT NOT NULL,
                canonical_en TEXT NOT NULL,
                entity_type TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                alias_text TEXT NOT NULL,
                dialect_region TEXT DEFAULT 'standard',
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS relations (
                src_id INTEGER NOT NULL,
                rel_type TEXT NOT NULL,
                dst_id INTEGER NOT NULL,
                provenance TEXT DEFAULT '',
                PRIMARY KEY (src_id, rel_type, dst_id),
                FOREIGN KEY (src_id) REFERENCES entities(id),
                FOREIGN KEY (dst_id) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_aliases_text
                ON aliases(alias_text COLLATE NOCASE);

            CREATE INDEX IF NOT EXISTS idx_entities_type
                ON entities(entity_type);

            CREATE INDEX IF NOT EXISTS idx_relations_src
                ON relations(src_id);

            CREATE INDEX IF NOT EXISTS idx_relations_dst
                ON relations(dst_id);
        """)
        self.conn.commit()
        logger.info("Knowledge graph schema initialized at %s", self.db_path)

    # --- CRUD Operations ---

    def add_entity(
        self,
        canonical_bn: str,
        canonical_en: str,
        entity_type: str,
    ) -> int:
        """Add an entity, return its ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO entities (canonical_bn, canonical_en, entity_type) VALUES (?, ?, ?)",
            (canonical_bn, canonical_en, entity_type),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_alias(
        self,
        entity_id: int,
        alias_text: str,
        dialect_region: str = "standard",
    ) -> int:
        """Add a dialect alias for an entity."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO aliases (entity_id, alias_text, dialect_region) VALUES (?, ?, ?)",
            (entity_id, alias_text, dialect_region),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_relation(
        self,
        src_id: int,
        rel_type: str,
        dst_id: int,
        provenance: str = "",
    ) -> None:
        """Add a directed relation between entities."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO relations (src_id, rel_type, dst_id, provenance) VALUES (?, ?, ?, ?)",
            (src_id, rel_type, dst_id, provenance),
        )
        self.conn.commit()

    # --- Query Operations ---

    def find_by_alias(self, text: str) -> list[Entity]:
        """Find entities matching an alias (case-insensitive)."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT e.id, e.canonical_bn, e.canonical_en, e.entity_type
            FROM entities e
            JOIN aliases a ON e.id = a.entity_id
            WHERE a.alias_text = ? COLLATE NOCASE
        """,
            (text,),
        )
        return [Entity(**dict(row)) for row in cursor.fetchall()]

    def find_by_partial_alias(self, text: str) -> list[Entity]:
        """Find entities with aliases containing the text."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT e.id, e.canonical_bn, e.canonical_en, e.entity_type
            FROM entities e
            JOIN aliases a ON e.id = a.entity_id
            WHERE a.alias_text LIKE ? COLLATE NOCASE
        """,
            (f"%{text}%",),
        )
        return [Entity(**dict(row)) for row in cursor.fetchall()]

    def get_entity(self, entity_id: int) -> Entity | None:
        """Get entity by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, canonical_bn, canonical_en, entity_type FROM entities WHERE id = ?",
            (entity_id,),
        )
        row = cursor.fetchone()
        return Entity(**dict(row)) if row else None

    def get_aliases(self, entity_id: int) -> list[Alias]:
        """Get all aliases for an entity."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, entity_id, alias_text, dialect_region FROM aliases WHERE entity_id = ?",
            (entity_id,),
        )
        return [Alias(**dict(row)) for row in cursor.fetchall()]

    def get_neighbors(
        self,
        entity_id: int,
        hops: int = 1,
        rel_types: list[str] | None = None,
    ) -> list[tuple[Entity, str]]:
        """
        Get neighboring entities within `hops` distance.

        Returns list of (Entity, relation_type) tuples.
        """
        visited = {entity_id}
        current_ids = {entity_id}
        results: list[tuple[Entity, str]] = []

        for _ in range(hops):
            next_ids = set()
            for eid in current_ids:
                cursor = self.conn.cursor()

                # Outgoing edges
                query = "SELECT dst_id, rel_type FROM relations WHERE src_id = ?"
                params: list = [eid]
                if rel_types:
                    placeholders = ",".join("?" * len(rel_types))
                    query += f" AND rel_type IN ({placeholders})"
                    params.extend(rel_types)

                cursor.execute(query, params)
                for row in cursor.fetchall():
                    neighbor_id = row["dst_id"]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_ids.add(neighbor_id)
                        entity = self.get_entity(neighbor_id)
                        if entity:
                            results.append((entity, row["rel_type"]))

                # Incoming edges
                query = "SELECT src_id, rel_type FROM relations WHERE dst_id = ?"
                params = [eid]
                if rel_types:
                    placeholders = ",".join("?" * len(rel_types))
                    query += f" AND rel_type IN ({placeholders})"
                    params.extend(rel_types)

                cursor.execute(query, params)
                for row in cursor.fetchall():
                    neighbor_id = row["src_id"]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_ids.add(neighbor_id)
                        entity = self.get_entity(neighbor_id)
                        if entity:
                            results.append((entity, row["rel_type"]))

            current_ids = next_ids

        return results

    def get_stats(self) -> dict:
        """Get KG statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        entity_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM aliases")
        alias_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM relations")
        relation_count = cursor.fetchone()[0]
        return {
            "entities": entity_count,
            "aliases": alias_count,
            "relations": relation_count,
        }

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
