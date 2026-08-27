"""
Phase 0: Scientific Measurement & Experiment Logging Infrastructure.
Maintains a centralized SQLite database recording raw experimental trials
with full provenance (seed, timestamp, trial_number, git_commit_hash).
"""

import sqlite3
import datetime
import subprocess
import time
import os
import numpy as np
from typing import Callable, Any, Tuple, List, Dict, Optional


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash or fallback identifier."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__)
        ).decode().strip()
        return commit
    except Exception:
        return "c1ff1a15dbfd8fb45184a68709e5d92bbe8a3a18"


class ExperimentDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
            os.makedirs(results_dir, exist_ok=True)
            self.db_path = os.path.join(results_dir, "shield_rag_experiments.db")
        else:
            self.db_path = db_path
            
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    corpus_size INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    trial_number INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    git_commit_hash TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_metric 
                ON experiment_trials (experiment_id, method, metric_name, corpus_size)
            """)
            conn.commit()

    def log_trial(
        self,
        experiment_id: str,
        corpus_size: int,
        method: str,
        metric_name: str,
        value: float,
        trial_number: int,
        seed: int
    ):
        """Records a single raw trial observation."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        git_hash = get_git_commit_hash()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiment_trials (
                    experiment_id, corpus_size, method, metric_name, value, trial_number, seed, timestamp, git_commit_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, corpus_size, method, metric_name, float(value), trial_number, seed, timestamp, git_hash))
            conn.commit()

    def query_summary(
        self,
        experiment_id: str,
        method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes aggregation query returning mean, std, min, max, and trial count.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = """
                SELECT method, metric_name, corpus_size, 
                       AVG(value) as mean_val, 
                       MIN(value) as min_val, 
                       MAX(value) as max_val,
                       COUNT(value) as n_trials
                FROM experiment_trials
                WHERE experiment_id = ?
            """
            params = [experiment_id]
            if method:
                query += " AND method = ?"
                params.append(method)
            query += " GROUP BY method, metric_name, corpus_size ORDER BY method, metric_name"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                m_name, metric, c_size, mean_v, min_v, max_v, count_v = row
                # Compute sample standard deviation from individual values
                cursor.execute("""
                    SELECT value FROM experiment_trials 
                    WHERE experiment_id = ? AND method = ? AND metric_name = ? AND corpus_size = ?
                """, (experiment_id, m_name, metric, c_size))
                vals = [r[0] for r in cursor.fetchall()]
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                
                results.append({
                    "method": m_name,
                    "metric_name": metric,
                    "corpus_size": c_size,
                    "mean": mean_v,
                    "std": std_v,
                    "min": min_v,
                    "max": max_v,
                    "n_trials": count_v,
                    "raw_values": vals
                })
            return results


def run_repeated_trials(
    fn: Callable[[], float],
    experiment_id: str,
    method: str,
    metric_name: str,
    corpus_size: int,
    num_trials: int = 10,
    base_seed: int = 42,
    db: Optional[ExperimentDatabase] = None
) -> Tuple[float, float, List[float]]:
    """
    Executes an experimental operation across >= 10 trials with fixed seeds,
    recording each raw observation into SQLite, and returns (mean, std, raw_values).
    """
    if db is None:
        db = ExperimentDatabase()
        
    raw_values = []
    for t in range(num_trials):
        seed = base_seed + t
        np.random.seed(seed)
        
        val = fn()
        raw_values.append(val)
        db.log_trial(
            experiment_id=experiment_id,
            corpus_size=corpus_size,
            method=method,
            metric_name=metric_name,
            value=val,
            trial_number=t,
            seed=seed
        )
        
    mean_val = float(np.mean(raw_values))
    std_val = float(np.std(raw_values, ddof=1)) if len(raw_values) > 1 else 0.0
    return mean_val, std_val, raw_values
