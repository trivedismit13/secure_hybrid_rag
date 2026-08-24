"""
SHIELD-RAG Ontology Schema — FROZEN after initial commit.

Defines the closed-set node types and relation types that all components depend on.
Component C's decoy pool logic requires a closed-set typing system; do not extend
these enums after Phase 3 begins without logging the change as a spec deviation.

This module is a candidate patent claim element — its interface, inputs, and outputs
must be documented precisely and versioned.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    """Closed-set ontology node types for the knowledge graph.

    These labels map directly to IC-HRAG's semantic-label design
    (Requirement / Activity(Action) / Block / Parameter) and drive
    Component C's type-cluster-based decoy pool selection.
    """

    REQUIREMENT = "Requirement"
    ACTION = "Action"
    BLOCK = "Block"
    PARAMETER = "Parameter"


class RelationType(str, Enum):
    """Closed-set relation types between knowledge graph nodes.

    Each relation constrains which (src_type, dst_type) pairs are valid,
    enabling intent-driven expansion filtering in the retrieval phase.
    """

    SATISFY = "Satisfy"            # Block -> Requirement
    TRACE = "Trace"                # Requirement -> Action
    ALLOCATE = "Allocate"          # Action -> Block
    HAS_PARAMETER = "HasParameter" # Block -> Parameter
    PART_OF = "PartOf"             # Block -> Block


# Valid (source_type, relation, destination_type) combinations.
# Used by the intent classifier and constrained expansion to prune invalid traversals.
VALID_RELATION_SCHEMA: dict[RelationType, tuple[NodeType, NodeType]] = {
    RelationType.SATISFY: (NodeType.BLOCK, NodeType.REQUIREMENT),
    RelationType.TRACE: (NodeType.REQUIREMENT, NodeType.ACTION),
    RelationType.ALLOCATE: (NodeType.ACTION, NodeType.BLOCK),
    RelationType.HAS_PARAMETER: (NodeType.BLOCK, NodeType.PARAMETER),
    RelationType.PART_OF: (NodeType.BLOCK, NodeType.BLOCK),
}


@dataclass
class GraphNode:
    """A node in the knowledge graph (plaintext representation).

    Attributes:
        node_id:   Unique identifier (UUID). Used only client-side before encryption.
        node_type: Ontology category from the closed-set NodeType enum.
        text:      Source text snippet this node represents.
        embedding: Dense vector (pre-encryption). Typically 384-dim from Sentence-BERT.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.BLOCK
    text: str = ""
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.node_type, str):
            self.node_type = NodeType(self.node_type)


@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph (plaintext representation).

    Attributes:
        src_id:   Source node UUID.
        dst_id:   Destination node UUID.
        relation: Relation type from the closed-set RelationType enum.
    """

    src_id: str = ""
    dst_id: str = ""
    relation: RelationType = RelationType.PART_OF

    def __post_init__(self) -> None:
        if isinstance(self.relation, str):
            self.relation = RelationType(self.relation)


@dataclass
class IntentLabel:
    """Output of Component A's intent classifier.

    Maps a user query to an ontology-constrained retrieval scope.

    Attributes:
        target_type:       The primary NodeType the query is asking about.
        allowed_relations: The set of RelationType values that the constrained
                           expansion is permitted to traverse from anchors.
        confidence:        Classifier confidence in [0, 1].
    """

    target_type: NodeType = NodeType.BLOCK
    allowed_relations: set[RelationType] = field(default_factory=set)
    confidence: float = 0.0


@dataclass
class RetrievedTriple:
    """A single (head, relation, tail) triple recovered during retrieval.

    Used as the atomic unit of context passed to the generation stage.
    """

    head: GraphNode = field(default_factory=GraphNode)
    relation: RelationType = RelationType.PART_OF
    tail: GraphNode = field(default_factory=GraphNode)
    score: float = 0.0  # relevance score from retrieval

    def to_text(self) -> str:
        """IC-HRAG-style knowledge-integration format string."""
        return f"({self.head.text} –{self.relation.value}→ {self.tail.text})"
