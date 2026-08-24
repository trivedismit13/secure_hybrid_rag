import hashlib
import json
from typing import Dict, List, Tuple, Any

class SimulatedContract:
    def __init__(self, oracle_address: str):
        self.owner = "0xOwnerAddress"
        self.oracle_address = oracle_address
        self.queries: Dict[bytes, Dict[str, Any]] = {}
        self.corpus_signatures: Dict[bytes, List[bytes]] = {}
        self.corpus_cids: Dict[bytes, List[str]] = {}
        self.events: List[Dict[str, Any]] = []

    def uploadCorpus(self, corpusId: bytes, signatures: List[bytes], cids: List[str]):
        """Equivalent to uploadCorpus Solidity function."""
        if len(signatures) != len(cids):
            raise ValueError("Solidity Revert: Mismatched inputs")
        self.corpus_signatures[corpusId] = signatures
        self.corpus_cids[corpusId] = cids
        self.events.append({
            'event': 'CorpusUploaded',
            'args': {
                'corpusId': corpusId.hex(),
                'count': len(signatures)
            }
        })

    def submitQuery(self, queryId: bytes, corpusId: bytes, sk_q: bytes, encryptedQueryHash: bytes):
        """Equivalent to submitQuery Solidity function."""
        self.queries[queryId] = {
            'sk_q': sk_q,
            'encryptedQueryHash': encryptedQueryHash,
            'exists': True,
            'processed': False,
            'topKCids': [],
            'auditHash': b'\x00' * 32
        }
        self.events.append({
            'event': 'QuerySubmitted',
            'args': {
                'queryId': queryId.hex(),
                'corpusId': corpusId.hex(),
                'sk_q': sk_q.hex(),
                'encryptedQueryHash': encryptedQueryHash.hex()
            }
        })

    def submitTopKResults(self, caller: str, queryId: bytes, topKCids: List[str], auditHash: bytes):
        """Equivalent to submitTopKResults Solidity function."""
        if caller != self.oracle_address:
            raise PermissionError("Solidity Revert: Only oracle can submit results")
        if queryId not in self.queries or not self.queries[queryId]['exists']:
            raise KeyError("Solidity Revert: Query does not exist")
        
        q = self.queries[queryId]
        q['processed'] = True
        q['topKCids'] = topKCids
        q['auditHash'] = auditHash
        self.events.append({
            'event': 'RetrievalCompleted',
            'args': {
                'queryId': queryId.hex(),
                'topKCids': topKCids,
                'auditHash': auditHash.hex()
            }
        })

    def getCorpus(self, corpusId: bytes) -> Tuple[List[bytes], List[str]]:
        """Equivalent to getCorpus view function."""
        return (self.corpus_signatures.get(corpusId, []), self.corpus_cids.get(corpusId, []))

    def getQueryResult(self, queryId: bytes) -> Tuple[bool, List[str], bytes]:
        """Equivalent to getQueryResult view function."""
        if queryId not in self.queries:
            raise KeyError("Solidity Revert: Query does not exist")
        q = self.queries[queryId]
        return (q['processed'], q['topKCids'], q['auditHash'])

class BlockchainSimulator:
    def __init__(self, oracle_address: str = "0xOracleAddress"):
        """
        Simulates local blockchain interaction (Ganache-like local EVM).
        Manages contract deployment, event subscription, and public audits.
        """
        self.oracle_address = oracle_address
        self.contract = SimulatedContract(self.oracle_address)
        self.accounts = ["0xOwnerAddress", self.oracle_address, "0xClientAddress"]

    def get_contract(self) -> SimulatedContract:
        return self.contract

    def get_events(self) -> List[Dict[str, Any]]:
        """Fetch all emitted event logs from the blockchain."""
        return self.contract.events

    def clear_events(self):
        self.contract.events = []
