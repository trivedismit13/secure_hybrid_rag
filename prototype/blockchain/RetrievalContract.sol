// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RetrievalContract {
    address public owner;
    address public oracleAddress;

    struct QueryRecord {
        bytes sk_q;
        bytes encryptedQueryHash;
        bool exists;
        bool processed;
        string[] topKCids;
        bytes32 auditHash;
    }

    // Mapping from queryId to QueryRecord
    mapping(bytes32 => QueryRecord) public queries;

    // Mapping from corpusId to array of encrypted database signatures
    mapping(bytes32 => bytes[]) public corpusSignatures;
    // Mapping from corpusId to corresponding IPFS CIDs
    mapping(bytes32 => string[]) public corpusCids;

    event CorpusUploaded(bytes32 indexed corpusId, uint256 count);
    event QuerySubmitted(bytes32 indexed queryId, bytes32 indexed corpusId, bytes sk_q, bytes encryptedQueryHash);
    event RetrievalCompleted(bytes32 indexed queryId, string[] topKCids, bytes32 auditHash);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyOracle() {
        require(msg.sender == oracleAddress, "Only oracle");
        _;
    }

    constructor(address _oracleAddress) {
        owner = msg.sender;
        oracleAddress = _oracleAddress;
    }

    function setOracleAddress(address _newOracle) external onlyOwner {
        oracleAddress = _newOracle;
    }

    function uploadCorpus(bytes32 corpusId, bytes[] calldata _signatures, string[] calldata _cids) external {
        require(_signatures.length == _cids.length, "Mismatched inputs");
        corpusSignatures[corpusId] = _signatures;
        corpusCids[corpusId] = _cids;
        emit CorpusUploaded(corpusId, _signatures.length);
    }

    // Client calls this to request retrieval
    function submitQuery(bytes32 queryId, bytes32 corpusId, bytes calldata sk_q, bytes calldata encryptedQueryHash) external {
        queries[queryId] = QueryRecord({
            sk_q: sk_q,
            encryptedQueryHash: encryptedQueryHash,
            exists: true,
            processed: false,
            topKCids: new string[](0),
            auditHash: bytes32(0)
        });
        emit QuerySubmitted(queryId, corpusId, sk_q, encryptedQueryHash);
    }

    // Oracle (Off-chain worker) calls this after running retrieval algorithm
    function submitTopKResults(bytes32 queryId, string[] calldata _topKCids, bytes32 _auditHash) external onlyOracle {
        require(queries[queryId].exists, "Query does not exist");
        QueryRecord storage q = queries[queryId];
        q.processed = true;
        q.topKCids = _topKCids;
        q.auditHash = _auditHash;
        emit RetrievalCompleted(queryId, _topKCids, _auditHash);
    }

    function getCorpus(bytes32 corpusId) external view returns (bytes[] memory, string[] memory) {
        return (corpusSignatures[corpusId], corpusCids[corpusId]);
    }

    function getQueryResult(bytes32 queryId) external view returns (bool processed, string[] memory topKCids, bytes32 auditHash) {
        require(queries[queryId].exists, "Query does not exist");
        QueryRecord storage q = queries[queryId];
        return (q.processed, q.topKCids, q.auditHash);
    }
}
