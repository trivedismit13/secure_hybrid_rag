# Formal Indistinguishability Proof for the Sensitivity-Embedded IPFE (SE-IPFE) Noise Gate

## 1. Claim Statement

> **Claim:** For any probabilistic polynomial-time adversary $\mathcal{A}$ (here, the semi-trusted Oracle) that does not know the clearance gap $(L_d - L_c)$, the distribution of $D = g^{\langle x, y \rangle + r \cdot (L_d - L_c)} \bmod N^2$ when $L_d > L_c$ is computationally indistinguishable from a uniformly random element of the Paillier ciphertext group, given that the Decisional Composite Residuosity (DCR) assumption holds for modulus $N$.

---

## 2. Security Game (IND-CPA-Style Game Structure)

1. **Setup:** The Key Distribution Center (KDC) runs Paillier parameter setup $\text{Setup}(1^\lambda)$, generating safe prime products $N = p \cdot q$ where $p = 2p' + 1$ and $q = 2q' + 1$, modulus $N^2$, generator $g = (1 + N) \bmod N^2$, and Carmichael function $\lambda(N) = \text{lcm}(p-1, q-1) = 2p'q'$. The KDC generates system blenders $\alpha, \beta \xleftarrow{R} \mathbb{Z}_{\lambda(N)}^*$ and publishes public parameters $\text{mpk} = \{N, N^2, g, \lambda(N), \text{pk}_1 = g^\alpha \bmod N^2, \text{pk}_2 = g^\beta \bmod N^2\}$ while keeping private factors $p, q$ and blenders $\alpha, \beta$ secret. The adversary $\mathcal{A}$ is provided only the public parameters $\text{mpk}$.

2. **Challenge:** The adversary $\mathcal{A}$ submits target document vector $x^* \in \mathbb{R}^d$, target query vector $y^* \in \mathbb{R}^d$, and two candidate clearance gaps $\delta_0 = (L_{d,0} - L_{c,0}) > 0$ and $\delta_1 = (L_{d,1} - L_{c,1}) > 0$ with $\delta_0, \delta_1 \in \{1, 2, 3, 4\}$. The challenger samples a uniform random coin $b \xleftarrow{R} \{0, 1\}$, draws a fresh blinding factor $r \xleftarrow{R} \mathbb{Z}_{\lambda(N)}^*$, computes the algebraic noise $\Delta_b = r \cdot \delta_b \bmod \lambda(N)$, derives the noised subkey $sk_{y,b} = (\langle s, y^* \rangle + \alpha + \beta + \Delta_b) \bmod \lambda(N)$, and computes the evaluated ciphertext inner product state:
   $$D_b = g^{\langle x^*, y^* \rangle - \Delta_b} \cdot w^N \bmod N^2 = g^{\langle x^*, y^* \rangle - r \cdot \delta_b} \cdot w^N \bmod N^2$$
   where $w \xleftarrow{R} \mathbb{Z}_N^*$. The challenger transmits challenge ciphertext $D_b$ to adversary $\mathcal{A}$.

3. **Guess:** The adversary $\mathcal{A}$ performs arbitrary polynomial-time computations over $D_b$ and outputs a bit guess $b' \in \{0, 1\}$.

4. **Advantage Definition:** The advantage of adversary $\mathcal{A}$ in distinguishing which denial gap $\delta_b$ was noised is formally defined as:
   $$\text{Adv}_{\mathcal{A}}^{\text{SE-IPFE-IND}}(\lambda) = \left| \Pr[b' = b] - \frac{1}{2} \right|$$

5. **Goal:** Show that $\text{Adv}_{\mathcal{A}}^{\text{SE-IPFE-IND}}(\lambda)$ is negligible in the security parameter $\lambda$ (the bit length of $N$), assuming the hardness of the Decisional Composite Residuosity (DCR) problem.

---

## 3. Reduction to Decisional Composite Residuosity (DCR)

### 3.1 The DCR Assumption
Let $N = pq$ be an RSA modulus. An element $z \in \mathbb{Z}_{N^2}^*$ is an $N$-th residue modulo $N^2$ if there exists $y \in \mathbb{Z}_{N^2}^*$ such that $z = y^N \bmod N^2$. The Decisional Composite Residuosity (DCR) assumption states that for any probabilistic polynomial-time algorithm $\mathcal{D}$, the distinguishing advantage:
$$\text{Adv}_{\mathcal{D}}^{\text{DCR}}(\lambda) = \left| \Pr[\mathcal{D}(N, z) = 1 \mid z \xleftarrow{R} \mathcal{R}es(N^2)] - \Pr[\mathcal{D}(N, z) = 1 \mid z \xleftarrow{R} \mathbb{Z}_{N^2}^*] \right|$$
is a negligible function $\text{negl}(\lambda)$ in security parameter $\lambda$.

### 3.2 Simulator Construction
We construct a polynomial-time simulator $\mathcal{S}$ that uses adversary $\mathcal{A}$ as a subroutine to solve the DCR problem. 

Simulator $\mathcal{S}$ receives an instance $(N, w)$ from the DCR challenger, where $w$ is either a random $N$-th residue $y^N \bmod N^2$ or a uniformly random element in $\mathbb{Z}_{N^2}^*$.

1. $\mathcal{S}$ sets up the public parameters $\text{mpk} = \{N, N^2, g = 1 + N \bmod N^2\}$ and chooses random blenders $\alpha, \beta \xleftarrow{R} [1, N]$.
2. When adversary $\mathcal{A}$ outputs challenge vectors $x^*, y^*$ and denial gaps $\delta_0, \delta_1$:
   - $\mathcal{S}$ flips an internal random coin $b \xleftarrow{R} \{0, 1\}$.
   - $\mathcal{S}$ chooses random $r' \xleftarrow{R} \mathbb{Z}_N^*$.
   - $\mathcal{S}$ embeds the DCR challenge element $w$ into the noise construction by computing:
     $$D_b = (1 + N)^{\langle x^*, y^* \rangle - r' \cdot \delta_b} \cdot w \bmod N^2$$
3. $\mathcal{S}$ hands $D_b$ to $\mathcal{A}$.
4. When $\mathcal{A}$ returns bit guess $b'$, $\mathcal{S}$ outputs 1 (guessing $w$ is an $N$-th residue) if $b' = b$, and outputs 0 (guessing $w$ is uniform) otherwise.

### 3.3 Analysis of the Simulator
- **Case 1: $w$ is an $N$-th residue ($w = y_0^N \bmod N^2$):**  
  In this case, $w$ is an exact Paillier randomizer. Since $r'$ is uniformly distributed in $\mathbb{Z}_N^*$ and $\gcd(\delta_b, N) = 1$ (since $\delta_b \in \{1, 2, 3, 4\} < p, q$), the product $\Delta_b = r' \cdot \delta_b \bmod N$ is uniformly distributed in $\mathbb{Z}_N$. Therefore, $D_b$ has the identical probability distribution to the legitimate challenge ciphertext in the real security game for choice $b$. Thus:
  $$\Pr[\mathcal{S}(N, w) = 1 \mid w \in \mathcal{R}es(N^2)] = \Pr[b' = b] = \frac{1}{2} + \text{Adv}_{\mathcal{A}}^{\text{SE-IPFE-IND}}(\lambda)$$

- **Case 2: $w$ is uniformly distributed in $\mathbb{Z}_{N^2}^*$:**  
  By Paillier's structural isomorphism $\mathbb{Z}_{N^2}^* \cong \mathbb{Z}_N \times \mathbb{Z}_N^*$, any uniformly random element $w \in \mathbb{Z}_{N^2}^*$ can be uniquely written as $(1 + N)^m \cdot y^N \bmod N^2$ where $m \in \mathbb{Z}_N$ is uniform and independent of $b$. Consequently, $D_b = (1 + N)^{\langle x^*, y^* \rangle - r'\delta_b + m} \cdot y^N \bmod N^2$ is perfectly uniform and statistically independent of bit $b$. Therefore, $\mathcal{A}$ has zero information about $b$:
  $$\Pr[\mathcal{S}(N, w) = 1 \mid w \xleftarrow{R} \mathbb{Z}_{N^2}^*] = \frac{1}{2}$$

### 3.4 Conclusion
Subtracting the probabilities yields:
$$\text{Adv}_{\mathcal{S}}^{\text{DCR}}(\lambda) = \left| \left( \frac{1}{2} + \text{Adv}_{\mathcal{A}}^{\text{SE-IPFE-IND}}(\lambda) \right) - \frac{1}{2} \right| = \text{Adv}_{\mathcal{A}}^{\text{SE-IPFE-IND}}(\lambda)$$

Because $\text{Adv}_{\mathcal{S}}^{\text{DCR}}(\lambda) \le \text{negl}(\lambda)$ by the DCR assumption, it follows that $\text{Adv}_{\mathcal{A}}^{\text{SE-IPFE-IND}}(\lambda) \le \text{negl}(\lambda)$. This concludes the proof that the noised discrete logarithm output reveals negligible information regarding the underlying clearance gap.

---

## 4. Scope and Limitations

1. **Specific Threat Model:** This proof is a reduction for the specific security game defined in Section 2—namely, distinguishing between two nonzero clearance gaps $\delta_0 \neq \delta_1$ given only public parameters and noised evaluation output. It does not constitute a generic proof against active side-channel attacks or collusion between the Oracle and KDC.
2. **Empirical Sanity Check Scope:** The accompanying statistical evaluation harness (`scripts/empirical_indistinguishability_test.py`) tests non-cryptographic statistical distinguishers (e.g., modular biases, bit lengths, and residue distributions). It validates the absence of trivial numerical leakage in the implementation, complementing rather than substituting for the theoretical DCR reduction above.
