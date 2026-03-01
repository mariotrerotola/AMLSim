"""Dataclass di configurazione per la simulazione AML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

VALID_FRAUD_TYPES = frozenset({
    "fan_in", "fan_out", "cycle", "bipartite", "stack",
    "random", "scatter_gather", "gather_scatter",
})


@dataclass
class SimulationConfig:
    """Tutti i parametri necessari per generare un dataset AML sintetico.

    Ogni campo ha un default ragionevole: basta specificare solo cio' che
    si vuole personalizzare.
    """

    # --- Generali ---
    num_accounts: int = 1000
    num_steps: int = 720
    base_date: str = "2017-01-01"
    seed: int = 0
    simulation_name: str = "sample"

    # --- Transazioni normali ---
    min_amount: float = 100.0
    max_amount: float = 1000.0
    min_balance: float = 50000.0
    max_balance: float = 100000.0

    # --- Pattern di frode ---
    num_fraud_patterns: int = 10
    fraud_types: List[str] = field(
        default_factory=lambda: ["fan_in", "fan_out", "cycle"]
    )
    fraud_min_amount: float = 100.0
    fraud_max_amount: float = 200.0
    fraud_min_accounts: int = 5
    fraud_max_accounts: int = 10

    # --- Output ---
    output_dir: str = "outputs"

    # --- Avanzati (raramente necessari) ---
    margin_ratio: float = 0.1
    degree_threshold: int = 10
    transaction_interval: int = 7
    num_branches: int = 1000

    def __post_init__(self) -> None:
        if self.num_accounts < 1:
            raise ValueError("num_accounts deve essere >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps deve essere >= 1")
        if self.min_amount > self.max_amount:
            raise ValueError("min_amount non puo' superare max_amount")
        if self.min_balance > self.max_balance:
            raise ValueError("min_balance non puo' superare max_balance")
        if self.fraud_min_amount > self.fraud_max_amount:
            raise ValueError("fraud_min_amount non puo' superare fraud_max_amount")
        if not 0.0 <= self.margin_ratio <= 1.0:
            raise ValueError("margin_ratio deve essere tra 0.0 e 1.0")
        for ft in self.fraud_types:
            if ft not in VALID_FRAUD_TYPES:
                raise ValueError(
                    f"Tipo di frode sconosciuto: '{ft}'. "
                    f"Validi: {sorted(VALID_FRAUD_TYPES)}"
                )

    def to_conf_dict(self, param_dir: str, tmp_dir: str) -> dict:
        """Costruisce il dizionario equivalente a ``conf.json``.

        Parameters
        ----------
        param_dir:
            Directory dove risiedono i file CSV di input generati.
        tmp_dir:
            Directory per i file temporanei intermedi.
        """
        return {
            "general": {
                "random_seed": self.seed,
                "simulation_name": self.simulation_name,
                "total_steps": self.num_steps,
                "base_date": self.base_date,
            },
            "default": {
                "min_amount": self.min_amount,
                "max_amount": self.max_amount,
                "min_balance": self.min_balance,
                "max_balance": self.max_balance,
                "start_step": -1,
                "end_step": -1,
                "start_range": -1,
                "end_range": -1,
                "transaction_model": 1,
                "margin_ratio": self.margin_ratio,
                "bank_id": "bank",
                "cash_in": {
                    "normal_interval": 100,
                    "fraud_interval": 50,
                    "normal_min_amount": 50,
                    "normal_max_amount": 100,
                    "fraud_min_amount": 500,
                    "fraud_max_amount": 1000,
                },
                "cash_out": {
                    "normal_interval": 10,
                    "fraud_interval": 100,
                    "normal_min_amount": 10,
                    "normal_max_amount": 100,
                    "fraud_min_amount": 1000,
                    "fraud_max_amount": 2000,
                },
            },
            "input": {
                "directory": param_dir,
                "schema": "schema.json",
                "accounts": "accounts.csv",
                "alert_patterns": "alertPatterns.csv",
                "normal_models": "normalModels.csv",
                "degree": "degree.csv",
                "transaction_type": "transactionType.csv",
                "is_aggregated_accounts": True,
            },
            "temporal": {
                "directory": tmp_dir,
                "transactions": "transactions.csv",
                "accounts": "accounts.csv",
                "alert_members": "alert_members.csv",
                "normal_models": "normal_models.csv",
            },
            "output": {
                "directory": self.output_dir,
                "accounts": "accounts.csv",
                "transactions": "transactions.csv",
                "cash_transactions": "cash_tx.csv",
                "alert_members": "alert_accounts.csv",
                "alert_transactions": "alert_transactions.csv",
                "sar_accounts": "sar_accounts.csv",
                "party_individuals": "individuals-bulkload.csv",
                "party_organizations": "organizations-bulkload.csv",
                "account_mapping": "accountMapping.csv",
                "resolved_entities": "resolvedentities.csv",
                "transaction_log": "tx_log.csv",
                "counter_log": "tx_count.csv",
                "diameter_log": "diameter.csv",
            },
            "graph_generator": {
                "degree_threshold": self.degree_threshold,
                "high_risk_countries": "",
                "high_risk_business": "",
            },
            "simulator": {
                "compute_diameter": False,
                "transaction_limit": 0,
                "transaction_interval": self.transaction_interval,
                "sar_interval": 7,
                "sar_balance_ratio": 1.0,
                "numBranches": self.num_branches,
            },
            "visualizer": {
                "degree": "deg.png",
                "wcc": "wcc.png",
                "alert": "alert.png",
                "count": "count.png",
                "clustering": "cc.png",
                "diameter": "diameter.png",
            },
        }
