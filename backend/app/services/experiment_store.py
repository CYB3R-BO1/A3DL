from __future__ import annotations

import sqlite3
from pathlib import Path


class ExperimentStore:
    def __init__(self, db_path: str = "../artifacts/a3dl.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    attack_type TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    epsilon REAL NOT NULL,
                    alpha REAL NOT NULL,
                    steps INTEGER NOT NULL,
                    clean_accuracy REAL NOT NULL,
                    adversarial_accuracy REAL NOT NULL,
                    attack_success_rate REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def upsert_experiment(self, row: tuple) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    run_id, attack_type, dataset, model_name, epsilon, alpha, steps,
                    clean_accuracy, adversarial_accuracy, attack_success_rate, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    attack_type=excluded.attack_type,
                    dataset=excluded.dataset,
                    model_name=excluded.model_name,
                    epsilon=excluded.epsilon,
                    alpha=excluded.alpha,
                    steps=excluded.steps,
                    clean_accuracy=excluded.clean_accuracy,
                    adversarial_accuracy=excluded.adversarial_accuracy,
                    attack_success_rate=excluded.attack_success_rate,
                    created_at=excluded.created_at
                """,
                row,
            )

    def list_experiments(self, limit: int = 50):
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT run_id, attack_type, dataset, model_name, epsilon, alpha, steps,
                       clean_accuracy, adversarial_accuracy, attack_success_rate, created_at
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cur.fetchall()

    def list_experiment_dicts(self, limit: int = 50) -> list[dict]:
        rows = self.list_experiments(limit=limit)
        experiments = []
        for row in rows:
            experiments.append(
                {
                    "run_id": row[0],
                    "attack_type": row[1],
                    "dataset": row[2],
                    "model_name": row[3],
                    "epsilon": row[4],
                    "alpha": row[5],
                    "steps": row[6],
                    "clean_accuracy": row[7],
                    "adversarial_accuracy": row[8],
                    "attack_success_rate": row[9],
                    "created_at": row[10],
                }
            )
        return experiments
