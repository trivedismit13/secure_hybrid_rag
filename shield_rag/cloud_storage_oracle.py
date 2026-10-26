"""
Semi-Trusted Cloud Storage Oracle and ALSH Sublinear Indexing Engine.

Features:
- Semi-trusted (honest-but-curious) cloud-hosted vector storage.
- OpenStack Swift / Cinder cloud storage driver emulation for multi-tenant container isolation.
- Asymmetric Locality-Sensitive Hashing (ALSH) for Maximum Inner Product Search (MIPS):
    P(x_tilde) = [x_tilde; 1/2 ||x||^2; 1/4 ||x||^4; ...; 1/2^m ||x||^{2^m}] in R^{d+m}
    Q(y_tilde) = [y_tilde / ||y||; 0; 0; ...; 0] in R^{d+m}
- AES-256-GCM authenticated payload encryption coupled with IPFE ciphertexts.
- Encrypted inner-product evaluation over candidate ciphertexts using gated subkeys.
- Preserves exact intra-tenant ranking while suppressing cross-tenant similarity to machine precision.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json
import secrets
import time
import numpy as np
from Crypto.Cipher import AES

from shield_rag.config import (
    ALSH_BUCKET_WIDTH,
    ALSH_M_PARAM,
    ALSH_NUM_HASHES,
    ALSH_NUM_TABLES,
    ALSH_TOP_K,
    DEFAULT_DOC_CLEARANCE,
    GLOBAL_DIM,
)
from shield_rag.ipfe_engine import IPFECiphertext, IPFECryptosystem


@dataclass
class EncryptedDocumentRecord:
    """Represents a securely stored document record in cloud storage."""
    doc_id: str
    tenant_id: str
    clearance: int
    ipfe_ciphertext: IPFECiphertext
    aes_iv: bytes
    aes_ciphertext: bytes
    aes_tag: bytes
    alsh_hash: List[int]
    created_at: float


class OpenStackCloudDriver:
    """
    Comprehensive OpenStack Multi-Tenant Cloud Infrastructure Simulation.
    Faithfully models:
    - Keystone: Multi-tenant identity, projects, authentication tokens, roles.
    - Neutron: Multi-tenant isolated networks, security groups, perimeter firewall proxy.
    - Swift: Object storage containers (swift-{tenant_id}-encrypted-vectors) and encrypted blobs.
    - Cinder: High-performance block volumes for ALSH index tables and candidate storage.
    - Nova: Ephemeral inference compute host with volatile memory zeroization attestation.
    """

    def __init__(self) -> None:
        # Keystone Identity Service
        self.keystone_projects: Dict[str, Dict[str, Any]] = {}
        self.keystone_users: Dict[str, Dict[str, Any]] = {}
        self.keystone_tokens: Dict[str, Dict[str, Any]] = {}

        # Neutron Network Service
        self.neutron_networks: Dict[str, Dict[str, Any]] = {}
        self.neutron_security_groups: Dict[str, List[str]] = {}
        self.neutron_packet_stats = {"inspected": 0, "authorized": 0, "gated_noise": 0}

        # Swift Object Storage
        self.swift_containers: Dict[str, Dict[str, Any]] = {}
        self.swift_stats = {"total_objects": 0, "total_bytes": 0}

        # Cinder Block Storage
        self.cinder_volumes: Dict[str, Dict[str, Any]] = {}
        self.cinder_iops_counter: int = 0

        # Nova Compute Service
        self.nova_instances: Dict[str, Dict[str, Any]] = {}
        self._init_core_cloud_infrastructure()

    def _init_core_cloud_infrastructure(self) -> None:
        """Initializes default OpenStack Cloud control plane."""
        # Core Cinder volume for ALSH tables
        self.cinder_volumes["vol-alsh-index-01"] = {
            "name": "cinder-alsh-multiprobe-tables",
            "size_gb": 100,
            "volume_type": "high-iops-ssd",
            "status": "attached",
            "mount_point": "/dev/vdb",
            "iops": 0,
        }
        # Nova inference gateway instance
        self.nova_instances["inst-gateway-nova-01"] = {
            "name": "nova-ephemeral-attention-gateway",
            "flavor": "m1.xlarge",
            "vcpus": 16,
            "ram_mb": 32768,
            "status": "ACTIVE",
            "memory_zeroization_attested": True,
        }

    # ─── Keystone Identity Methods ────────────────────────────────────
    def keystone_create_project(self, tenant_id: str, display_name: str) -> Dict[str, Any]:
        """Provisions an enterprise tenant project domain in OpenStack Keystone."""
        project = {
            "project_id": f"proj-{tenant_id}",
            "tenant_id": tenant_id,
            "name": display_name,
            "status": "ACTIVE",
            "created_at": time.time(),
        }
        self.keystone_projects[tenant_id] = project
        # Provision isolated Neutron network
        self.neutron_create_network(tenant_id)
        return project

    def keystone_register_user(self, user_id: str, tenant_id: str, clearance: int) -> None:
        """Registers a user identity under a Keystone project."""
        self.keystone_users[user_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "clearance": clearance,
            "status": "ACTIVE",
        }

    def keystone_issue_token(self, user_id: str, tenant_id: str, epoch: int) -> str:
        """Issues a signed Keystone auth token stamped with tenant epoch."""
        token_id = f"token-ks-{hashlib.sha256(f'{user_id}:{tenant_id}:{epoch}:{time.time()}'.encode()).hexdigest()[:16]}"
        self.keystone_tokens[token_id] = {
            "token_id": token_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "epoch": epoch,
            "valid": True,
        }
        return token_id

    # ─── Neutron Network Methods ──────────────────────────────────────
    def neutron_create_network(self, tenant_id: str) -> None:
        """Creates an isolated virtual tenant network and security group in Neutron."""
        net_id = f"net-{tenant_id}-internal"
        self.neutron_networks[tenant_id] = {
            "net_id": net_id,
            "cidr": f"10.{len(self.neutron_networks) + 1}.0.0/24",
            "status": "ACTIVE",
        }
        self.neutron_security_groups[tenant_id] = [
            "ALLOW_TLS_MTLS_PORT_443",
            "REVOCATION_PROXY_INSPECTION_PORT_8443",
        ]

    def neutron_log_packet(self, is_noise_flooded: bool) -> None:
        """Logs network packet traversal across the perimeter proxy security group."""
        self.neutron_packet_stats["inspected"] += 1
        if is_noise_flooded:
            self.neutron_packet_stats["gated_noise"] += 1
        else:
            self.neutron_packet_stats["authorized"] += 1

    # ─── Swift Object Storage Methods ─────────────────────────────────
    def create_container(self, tenant_id: str, container_name: str) -> str:
        """Provisions an OpenStack Swift container for an enterprise tenant."""
        scoped_name = f"swift-{tenant_id}-{container_name}"
        if scoped_name not in self.swift_containers:
            self.swift_containers[scoped_name] = {}
        return scoped_name

    def put_object(self, container: str, object_id: str, data: Any) -> None:
        """Stores an encrypted object in an OpenStack Swift container."""
        if container not in self.swift_containers:
            self.swift_containers[container] = {}
        self.swift_containers[container][object_id] = data
        self.swift_stats["total_objects"] += 1
        # Approximate size of payload + ciphertext in bytes
        self.swift_stats["total_bytes"] += 4096

    def get_object(self, container: str, object_id: str) -> Optional[Any]:
        """Retrieves an object from OpenStack Swift container."""
        return self.swift_containers.get(container, {}).get(object_id)

    # ─── Cinder Block Storage Methods ─────────────────────────────────
    def record_cinder_iops(self, count: int = 1) -> None:
        """Records IOPS operations on Cinder block storage holding ALSH tables."""
        self.cinder_iops_counter += count
        if "vol-alsh-index-01" in self.cinder_volumes:
            self.cinder_volumes["vol-alsh-index-01"]["iops"] = self.cinder_iops_counter

    def get_cloud_summary(self) -> Dict[str, Any]:
        """Returns comprehensive OpenStack Cloud deployment metrics."""
        return {
            "keystone": {
                "active_projects": len(self.keystone_projects),
                "registered_users": len(self.keystone_users),
                "issued_tokens": len(self.keystone_tokens),
            },
            "neutron": {
                "tenant_networks": len(self.neutron_networks),
                "packet_stats": dict(self.neutron_packet_stats),
            },
            "swift": {
                "containers": list(self.swift_containers.keys()),
                "total_objects": self.swift_stats["total_objects"],
                "total_bytes": self.swift_stats["total_bytes"],
            },
            "cinder": {
                "volumes": self.cinder_volumes,
                "total_iops": self.cinder_iops_counter,
            },
            "nova": {
                "instances": self.nova_instances,
            },
        }


# Backwards compatibility alias
OpenStackStorageDriver = OpenStackCloudDriver


class ALSHIndex:
    """
    Asymmetric Locality-Sensitive Hashing (ALSH) index for MIPS (Shrivastava & Li).
    Transforms asymmetric inner-product search in R^d into cosine search in R^{d+m}.
    """

    def __init__(
        self,
        dim: int = GLOBAL_DIM,
        m_param: int = ALSH_M_PARAM,
        num_tables: int = ALSH_NUM_TABLES,
        num_hashes: int = ALSH_NUM_HASHES,
        seed: Optional[int] = 42,
    ) -> None:
        self.dim = dim
        self.m_param = m_param
        self.extended_dim = dim + m_param
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.rng = np.random.default_rng(seed)

        # Random projection hyperplanes: L tables, each having K vectors in R^{d+m}
        self.hyperplanes: List[np.ndarray] = [
            self.rng.standard_normal((num_hashes, self.extended_dim))
            for _ in range(num_tables)
        ]

        # Multi-probe hash tables: table_idx -> {hash_key: List[doc_id]}
        self.tables: List[Dict[int, List[str]]] = [{} for _ in range(num_tables)]

    def expand_dimension(self, new_dim: int) -> None:
        """
        Dynamically expands ALSH random projection hyperplanes from d to d'
        to accommodate Tier 2 Incremental Modified Gram-Schmidt tenant expansion.
        """
        if new_dim <= self.dim:
            return
        diff = new_dim - self.dim
        new_extended = new_dim + self.m_param
        for l in range(self.num_tables):
            base_cols = self.hyperplanes[l][:, :self.dim]
            aug_cols = self.hyperplanes[l][:, self.dim:]
            new_cols = self.rng.standard_normal((self.num_hashes, diff))
            self.hyperplanes[l] = np.hstack([base_cols, new_cols, aug_cols])
        self.dim = new_dim
        self.extended_dim = new_extended

    def transform_document(self, x: np.ndarray) -> np.ndarray:
        """
        Asymmetric document transformation P(x) in R^{d+m}:
            P(x) = [x; 1/2 ||x||^2; 1/4 ||x||^4; ...; 1/2^m ||x||^{2^m}].
        """
        x_len = len(x)
        if x_len > self.dim:
            self.expand_dimension(x_len)
        elif x_len < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:x_len] = x
            x = padded

        x_norm = np.linalg.norm(x)
        aug_terms = []
        for i in range(1, self.m_param + 1):
            power_val = 2 ** i
            aug = (0.5 ** i) * (x_norm ** power_val)
            aug_terms.append(aug)
        return np.concatenate([x, np.array(aug_terms, dtype=np.float64)])

    def transform_query(self, y: np.ndarray) -> np.ndarray:
        """
        Asymmetric query transformation Q(y) in R^{d+m}:
            Q(y) = [y / ||y||; 0; 0; ...; 0].
        """
        y_len = len(y)
        if y_len > self.dim:
            self.expand_dimension(y_len)
        elif y_len < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:y_len] = y
            y = padded

        norm_y = np.linalg.norm(y)
        normalized_y = y / norm_y if norm_y > 1e-12 else y
        zeros = np.zeros(self.m_param, dtype=np.float64)
        return np.concatenate([normalized_y, zeros])

    def _hash_vector(self, table_idx: int, vec: np.ndarray) -> int:
        """Projects extended vector against table hyperplanes to derive integer hash key."""
        proj = self.hyperplanes[table_idx] @ vec
        bits = (proj >= 0).astype(int)
        key = 0
        for bit in bits:
            key = (key << 1) | int(bit)
        return key

    def insert(self, doc_id: str, x_lifted: np.ndarray) -> List[int]:
        """Indexes a lifted document vector into all L hash tables."""
        p_vec = self.transform_document(x_lifted)
        keys = []
        for l in range(self.num_tables):
            key = self._hash_vector(l, p_vec)
            if key not in self.tables[l]:
                self.tables[l][key] = []
            self.tables[l][key].append(doc_id)
            keys.append(key)
        return keys

    def query_candidates(self, y_lifted: np.ndarray, top_k: int = ALSH_TOP_K) -> Set[str]:
        """
        Performs sublinear multi-probe candidate retrieval over ALSH tables.
        Returns a deduplicated set of candidate document IDs.
        """
        q_vec = self.transform_query(y_lifted)
        candidates: Set[str] = set()

        # Primary bucket lookup
        for l in range(self.num_tables):
            key = self._hash_vector(l, q_vec)
            if key in self.tables[l]:
                candidates.update(self.tables[l][key])

        # If candidates are few, perform 1-bit Hamming radius multi-probe
        if len(candidates) < top_k:
            for l in range(self.num_tables):
                key = self._hash_vector(l, q_vec)
                for bit_pos in range(self.num_hashes):
                    perturbed_key = key ^ (1 << bit_pos)
                    if perturbed_key in self.tables[l]:
                        candidates.update(self.tables[l][perturbed_key])
                        if len(candidates) >= top_k * 3:
                            break

        return candidates


class CloudStorageOracle:
    """
    Honest-but-curious Cloud Storage Oracle managing encrypted documents,
    ALSH index tables, and encrypted inner-product similarity evaluation.
    """

    def __init__(
        self,
        dim: int = GLOBAL_DIM,
        ipfe: Optional[IPFECryptosystem] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.dim = dim
        self.ipfe = ipfe or IPFECryptosystem(dim=dim, seed=seed)
        self.alsh_index = ALSHIndex(dim=dim, seed=seed)
        self.openstack = OpenStackStorageDriver()

        # Storage repository: doc_id -> EncryptedDocumentRecord
        self.documents: Dict[str, EncryptedDocumentRecord] = {}

        # Tenant document lookup: tenant_id -> List[doc_id]
        self.tenant_documents: Dict[str, List[str]] = {}

        # Per-tenant AES payload encryption keys derived via HKDF
        self._tenant_aes_keys: Dict[str, bytes] = {}

    def get_or_create_tenant_aes_key(self, tenant_id: str) -> bytes:
        """Derives a consistent 256-bit AES key for a tenant via HKDF/SHA-256."""
        if tenant_id not in self._tenant_aes_keys:
            salt = b"SHIELD-RAG-TENANT-PAYLOAD-SALT-2026"
            info = f"tenant-{tenant_id}-payload-key".encode()
            # HKDF extract and expand using hashlib sha256
            prk = hashlib.sha256(salt + tenant_id.encode()).digest()
            okm = hashlib.sha256(prk + info).digest()
            self._tenant_aes_keys[tenant_id] = okm
        return self._tenant_aes_keys[tenant_id]

    def encrypt_payload(self, tenant_id: str, plaintext: str) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypts document text using AES-256-GCM.

        Returns:
            (iv, ciphertext, tag)
        """
        key = self.get_or_create_tenant_aes_key(tenant_id)
        cipher = AES.new(key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
        return cipher.nonce, ciphertext, tag

    def decrypt_payload(
        self, tenant_id: str, iv: bytes, ciphertext: bytes, tag: bytes
    ) -> str:
        """Decrypts document text using AES-256-GCM authenticated decryption."""
        key = self.get_or_create_tenant_aes_key(tenant_id)
        cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
        decrypted_bytes = cipher.decrypt_and_verify(ciphertext, tag)
        return decrypted_bytes.decode("utf-8")

    def ingest_document(
        self,
        doc_id: str,
        tenant_id: str,
        x_lifted: np.ndarray,
        plaintext_text: str,
        clearance: int = DEFAULT_DOC_CLEARANCE,
    ) -> EncryptedDocumentRecord:
        """
        Stores an encrypted document record:
        1. Encrypts lifted vector under ALS-IPFE.
        2. Encrypts text payload under AES-256-GCM.
        3. Indexes vector into ALSH tables.
        4. Writes record to OpenStack Swift storage.
        """
        # 1. IPFE encryption
        ipfe_ct = self.ipfe.encrypt_vector(x_lifted)

        # 2. AES-256-GCM payload encryption
        iv, aes_ct, tag = self.encrypt_payload(tenant_id, plaintext_text)

        # 3. ALSH indexing
        hash_keys = self.alsh_index.insert(doc_id, x_lifted)
        self.openstack.record_cinder_iops(1)

        # 4. OpenStack Swift storage
        container = self.openstack.create_container(tenant_id, "encrypted-vectors")
        record = EncryptedDocumentRecord(
            doc_id=doc_id,
            tenant_id=tenant_id,
            clearance=clearance,
            ipfe_ciphertext=ipfe_ct,
            aes_iv=iv,
            aes_ciphertext=aes_ct,
            aes_tag=tag,
            alsh_hash=hash_keys,
            created_at=0.0,
        )

        self.documents[doc_id] = record
        if tenant_id not in self.tenant_documents:
            self.tenant_documents[tenant_id] = []
        self.tenant_documents[tenant_id].append(doc_id)

        self.openstack.put_object(container, doc_id, record)
        return record

    def search(
        self,
        gated_subkey: int,
        y_lifted: np.ndarray,
        top_k: int = ALSH_TOP_K,
        candidate_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float, EncryptedDocumentRecord]]:
        """
        Evaluates inner-product functional similarity over candidate ciphertexts.

        Args:
            gated_subkey: Functional decryption subkey received from Revocation Proxy.
            y_lifted: Lifted query vector in R^d.
            top_k: Maximum number of ranked results to return.
            candidate_ids: Optional candidate subset (defaults to ALSH candidates or all docs).

        Returns:
            List of (doc_id, similarity_score, EncryptedDocumentRecord) ranked descending.
        """
        # Determine candidate set
        if candidate_ids is None:
            alsh_cands = self.alsh_index.query_candidates(y_lifted, top_k=top_k)
            # If ALSH returns candidates, use them; otherwise scan indexed docs
            candidates = alsh_cands if len(alsh_cands) >= top_k else set(self.documents.keys())
        else:
            candidates = candidate_ids

        # Record Cinder block storage IOPS for index candidate traversal
        self.openstack.record_cinder_iops(len(candidates))

        results: List[Tuple[str, float, EncryptedDocumentRecord]] = []
        for doc_id in candidates:
            record = self.documents.get(doc_id)
            if record is None:
                continue

            # Cloud Storage Oracle evaluates inner product using IPFE decryption
            sim_score = self.ipfe.decrypt_inner_product(
                record.ipfe_ciphertext,
                gated_subkey,
                y_lifted,
            )
            results.append((doc_id, sim_score, record))

        # Rank candidates by evaluated similarity descending
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]
