"""
Structured Prompt Builder (Component D).

Builds LLM prompts from retrieved graph nodes while mapping structural relation
tokens to their required cryptographic subkeys. This allows the attention hook
to selectively mask relations based on the client's intent subkeys.
"""

from typing import Dict, List, Tuple, Any
from shield_rag.schema.ontology import RelationType, GraphNode
from shield_rag.crypto.relation_subkeys import RelationKeyManager


class StructuredPromptBuilder:
    def __init__(self, tokenizer: Any, key_manager: RelationKeyManager):
        """
        Args:
            tokenizer: Hugging Face tokenizer instance used by the LLM.
            key_manager: Instance of RelationKeyManager to derive required subkeys.
        """
        self.tokenizer = tokenizer
        self.key_manager = key_manager
        
    def build_prompt(
        self, 
        query: str, 
        nodes: List[GraphNode], 
        triples: List[Tuple[str, RelationType, str]]
    ) -> Tuple[str, Dict[int, bytes]]:
        """
        Builds the prompt string and computes the token requirements map.
        
        Returns:
            prompt_str: The plaintext prompt to feed to the LLM.
            token_requirements: Map of token_index -> required_subkey.
        """
        # Base instruction
        prompt_parts = [
            "<|im_start|>system",
            "You are a strict technical analyst. Answer the user's question based strictly on the provided knowledge graph structure. The graph contains nodes and relationships.",
            "<|im_end|>",
            "<|im_start|>user",
            f"Question: {query}",
            "",
            "Knowledge Graph:",
        ]
        
        # Tokenizer tracking
        # We need to compute the exact token index of the relation tokens.
        # To do this reliably, we encode the prompt incrementally.
        
        current_prompt = "\n".join(prompt_parts) + "\n"
        base_tokens = self.tokenizer.encode(current_prompt, add_special_tokens=False)
        current_token_idx = len(base_tokens)
        
        token_requirements: Dict[int, bytes] = {}
        
        # Build node dictionary for text lookup
        node_map = {n.node_id: n.text for n in nodes}
        
        built_str = current_prompt
        
        for head_id, relation, tail_id in triples:
            head_text = node_map.get(head_id, f"Node_{head_id}")
            tail_text = node_map.get(tail_id, f"Node_{tail_id}")
            
            # Format: [head] --RELATION--> [tail]
            head_part = f"- [{head_text}] --"
            rel_part = f"{relation.value}"
            tail_part = f"--> [{tail_text}]\n"
            
            # Tokenize parts
            head_tokens = self.tokenizer.encode(head_part, add_special_tokens=False)
            rel_tokens = self.tokenizer.encode(rel_part, add_special_tokens=False)
            tail_tokens = self.tokenizer.encode(tail_part, add_special_tokens=False)
            
            # Update built string
            built_str += head_part + rel_part + tail_part
            
            # Update tracking
            current_token_idx += len(head_tokens)
            
            # The relation tokens must be masked if the key is missing
            required_key = self.key_manager.derive_subkey(relation)
            for i in range(len(rel_tokens)):
                token_requirements[current_token_idx + i] = required_key
                
            current_token_idx += len(rel_tokens)
            current_token_idx += len(tail_tokens)
            
        # Append ending
        ending = "<|im_end|>\n<|im_start|>assistant\nAnswer:\n"
        built_str += ending
        
        return built_str, token_requirements
