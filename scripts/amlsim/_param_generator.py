"""Generazione automatica dei file CSV di parametri per la simulazione."""

from __future__ import annotations

import csv
import json
import os
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amlsim._config import SimulationConfig

# Distribuzione dei gradi di default.
# Ogni tupla e' (percentuale, in-degree, out-degree).
# Usare in_deg == out_deg per ogni riga garantisce il bilanciamento.
_DEFAULT_DEGREE_DISTRIBUTION = [
    (0.30, 3, 3),
    (0.25, 5, 5),
    (0.18, 8, 8),
    (0.12, 12, 12),
    (0.08, 18, 18),
    (0.04, 25, 25),
    (0.03, 35, 35),
]

# Tipi di modelli normali con pesi relativi di default.
_NORMAL_MODEL_TYPES = [
    ("single", 1, 1),
    ("fan_out", 5, 20),
    ("fan_in", 5, 20),
    ("forward", 3, 3),
    ("mutual", 2, 2),
    ("periodical", 2, 2),
]


def _find_schema_json() -> str:
    """Trova il file schema.json nel progetto."""
    candidates = [
        os.path.join("paramFiles", "1K", "schema.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "paramFiles", "1K", "schema.json"),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    raise FileNotFoundError(
        "schema.json non trovato. Assicurati che paramFiles/1K/schema.json esista."
    )


def generate_param_files(config: SimulationConfig, param_dir: str) -> None:
    """Genera tutti i file CSV di parametri necessari nella directory indicata.

    Parameters
    ----------
    config:
        Configurazione della simulazione.
    param_dir:
        Directory di destinazione per i file generati.
    """
    os.makedirs(param_dir, exist_ok=True)

    _write_accounts_csv(config, param_dir)
    _write_alert_patterns_csv(config, param_dir)
    _write_normal_models_csv(config, param_dir)
    _write_degree_csv(config, param_dir)
    _write_transaction_type_csv(param_dir)
    _copy_schema_json(param_dir)


def _write_accounts_csv(config: SimulationConfig, param_dir: str) -> None:
    path = os.path.join(param_dir, "accounts.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "min_balance", "max_balance", "country", "business_type", "bank_id"])
        writer.writerow([
            config.num_accounts,
            int(config.min_balance),
            int(config.max_balance),
            "US",
            "I",
            "bank",
        ])


def _write_alert_patterns_csv(config: SimulationConfig, param_dir: str) -> None:
    path = os.path.join(param_dir, "alertPatterns.csv")
    num_types = len(config.fraud_types)
    if num_types == 0:
        # Nessun pattern di frode richiesto
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "count", "type", "schedule_id", "min_accounts", "max_accounts",
                "min_amount", "max_amount", "min_period", "max_period", "bank_id", "is_sar",
            ])
        return

    base_count = config.num_fraud_patterns // num_types
    remainder = config.num_fraud_patterns % num_types

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count", "type", "schedule_id", "min_accounts", "max_accounts",
            "min_amount", "max_amount", "min_period", "max_period", "bank_id", "is_sar",
        ])
        for i, fraud_type in enumerate(config.fraud_types):
            count = base_count + (1 if i < remainder else 0)
            if count == 0:
                continue
            writer.writerow([
                count,
                fraud_type,
                2,
                config.fraud_min_accounts,
                config.fraud_max_accounts,
                f"{config.fraud_min_amount:.1f}",
                f"{config.fraud_max_amount:.1f}",
                5,
                20,
                "bank",
                "True",
            ])


def _write_normal_models_csv(config: SimulationConfig, param_dir: str) -> None:
    path = os.path.join(param_dir, "normalModels.csv")
    # Ogni tipo di modello riceve una quota proporzionale al num_accounts.
    # Il count e' il numero di *gruppi* di quel tipo, non il numero di account.
    # Usiamo num_accounts // 2 come numero totale di modelli da distribuire.
    total_models = max(config.num_accounts // 2, len(_NORMAL_MODEL_TYPES))
    per_type = total_models // len(_NORMAL_MODEL_TYPES)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count", "type", "schedule_id", "min_accounts", "max_accounts",
            "min_period", "max_period", "bank_id",
        ])
        for model_type, min_accts, max_accts in _NORMAL_MODEL_TYPES:
            writer.writerow([per_type, model_type, 2, min_accts, max_accts, 5, 20, "bank"])


def _write_degree_csv(config: SimulationConfig, param_dir: str) -> None:
    """Scrive degree.csv con una distribuzione bilanciata.

    Usa in_deg == out_deg per ogni riga, garantendo automaticamente
    che la somma totale degli in-degree == somma degli out-degree.
    """
    path = os.path.join(param_dir, "degree.csv")
    n = config.num_accounts

    # Calcola i count basati sulle percentuali
    rows = []
    assigned = 0
    for i, (pct, in_deg, out_deg) in enumerate(_DEFAULT_DEGREE_DISTRIBUTION):
        if i == len(_DEFAULT_DEGREE_DISTRIBUTION) - 1:
            count = n - assigned  # L'ultima riga prende il resto
        else:
            count = max(round(n * pct), 0)
        assigned += count
        if count > 0:
            rows.append([count, in_deg, out_deg])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Count", "In-degree", "Out-degree"])
        for row in rows:
            writer.writerow(row)


def _write_transaction_type_csv(param_dir: str) -> None:
    path = os.path.join(param_dir, "transactionType.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Frequency"])
        writer.writerow(["TRANSFER", 1])


def _copy_schema_json(param_dir: str) -> None:
    src = _find_schema_json()
    dst = os.path.join(param_dir, "schema.json")
    shutil.copy2(src, dst)
