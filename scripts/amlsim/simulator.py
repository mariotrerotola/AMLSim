"""Classe AMLSim: interfaccia semplificata per generare dataset AML sintetici.

Esempio d'uso::

    from amlsim import AMLSim

    sim = AMLSim(num_accounts=1000, num_steps=720, seed=42)
    sim.run()

    transactions = sim.to_dataframe()
    sar_accounts = sim.get_sar_accounts()
    alerts = sim.get_alerts()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Optional

import pandas as pd

from amlsim._config import SimulationConfig
from amlsim._param_generator import generate_param_files

logger = logging.getLogger(__name__)


class AMLSim:
    """Simulatore AML semplificato per data scientist.

    Parameters
    ----------
    **kwargs:
        Qualsiasi parametro accettato da :class:`SimulationConfig`.
        Vedi la documentazione di ``SimulationConfig`` per la lista completa.

    Esempi
    ------
    >>> sim = AMLSim(num_accounts=500, num_steps=365, seed=42)
    >>> sim.run()
    >>> df = sim.to_dataframe()
    >>> print(df.shape)
    """

    def __init__(self, **kwargs) -> None:
        self.config = SimulationConfig(**kwargs)
        self._transactions_df: Optional[pd.DataFrame] = None
        self._sar_df: Optional[pd.DataFrame] = None
        self._alerts_df: Optional[pd.DataFrame] = None
        self._output_path: Optional[str] = None

    def run(self) -> "AMLSim":
        """Esegue l'intera pipeline: generazione grafo, simulazione, conversione.

        Returns
        -------
        self
            Per consentire il chaining: ``AMLSim(...).run().to_dataframe()``
        """
        # Assicuriamoci che il path degli script sia nel sys.path
        scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        tmp_base = tempfile.mkdtemp(prefix="amlsim_")
        param_dir = os.path.join(tmp_base, "params")
        tmp_dir = os.path.join(tmp_base, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            self._run_pipeline(param_dir, tmp_dir, tmp_base)
        finally:
            # Pulisci i file temporanei (ma non l'output)
            shutil.rmtree(tmp_base, ignore_errors=True)

        return self

    def _run_pipeline(self, param_dir: str, tmp_dir: str, tmp_base: str) -> None:
        config = self.config

        # 1. Genera i file di parametri
        logger.info("Generazione file di parametri...")
        generate_param_files(config, param_dir)

        # 2. Costruisci il conf dict
        conf = config.to_conf_dict(param_dir, tmp_dir)

        # Scrivi conf.json temporaneo (necessario per SimProperties)
        conf_path = os.path.join(tmp_base, "conf.json")
        with open(conf_path, "w") as f:
            json.dump(conf, f)

        # 3. Genera il grafo delle transazioni
        logger.info("Generazione grafo transazioni...")
        self._run_graph_generator(conf, config.simulation_name)

        # 4. Esegui la simulazione
        logger.info("Esecuzione simulazione...")
        self._run_simulation(conf_path, config.simulation_name)

        # 5. Converti i log
        logger.info("Conversione log...")
        self._run_log_converter(conf, config.simulation_name)

        # 6. Carica i risultati
        self._output_path = os.path.join(
            config.output_dir, config.simulation_name
        )
        self._load_results()

    def _run_graph_generator(self, conf: dict, sim_name: str) -> None:
        from transaction_graph_generator import TransactionGenerator

        txg = TransactionGenerator(conf, sim_name)
        txg.set_num_accounts()
        txg.generate_normal_transactions()
        txg.load_account_list()
        txg.load_normal_models()
        txg.build_normal_models()
        txg.set_main_acct_candidates()
        txg.load_alert_patterns()
        txg.mark_active_edges()
        txg.write_account_list()
        txg.write_transaction_list()
        txg.write_alert_account_list()
        txg.write_normal_models()

    def _run_simulation(self, conf_path: str, sim_name: str) -> None:
        from amlsim.sim_properties import SimProperties
        from amlsim.python_runtime import PythonAMLSim

        sim_properties = SimProperties(conf_path, sim_name)
        runtime = PythonAMLSim(sim_properties)
        runtime.execute()

    def _run_log_converter(self, conf: dict, sim_name: str) -> None:
        from convert_logs import LogConverter
        from faker import Faker

        Faker.seed(self.config.seed)
        fake = Faker(["en_US"])
        converter = LogConverter(conf, sim_name, fake)
        converter.convert_alert_members()
        converter.convert_acct_tx()
        converter.output_sar_cases()

    def _load_results(self) -> None:
        if self._output_path is None:
            return

        tx_path = os.path.join(self._output_path, "transactions.csv")
        if os.path.exists(tx_path):
            self._transactions_df = pd.read_csv(tx_path)

        sar_path = os.path.join(self._output_path, "sar_accounts.csv")
        if os.path.exists(sar_path):
            self._sar_df = pd.read_csv(sar_path)

        alert_path = os.path.join(self._output_path, "alert_transactions.csv")
        if os.path.exists(alert_path):
            self._alerts_df = pd.read_csv(alert_path)

    def to_dataframe(self) -> pd.DataFrame:
        """Restituisce le transazioni come DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame con le colonne delle transazioni generate.

        Raises
        ------
        RuntimeError
            Se ``run()`` non e' stato ancora chiamato.
        """
        if self._transactions_df is None:
            raise RuntimeError(
                "Nessun risultato disponibile. Chiama run() prima di to_dataframe()."
            )
        return self._transactions_df

    def get_sar_accounts(self) -> pd.DataFrame:
        """Restituisce i conti SAR (Suspicious Activity Report) come DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame con i conti sospetti.

        Raises
        ------
        RuntimeError
            Se ``run()`` non e' stato ancora chiamato.
        """
        if self._sar_df is None:
            raise RuntimeError(
                "Nessun risultato disponibile. Chiama run() prima di get_sar_accounts()."
            )
        return self._sar_df

    def get_alerts(self) -> pd.DataFrame:
        """Restituisce gli alert (transazioni sospette) come DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame con le transazioni di alert.

        Raises
        ------
        RuntimeError
            Se ``run()`` non e' stato ancora chiamato.
        """
        if self._alerts_df is None:
            raise RuntimeError(
                "Nessun risultato disponibile. Chiama run() prima di get_alerts()."
            )
        return self._alerts_df

    def __repr__(self) -> str:
        status = "completata" if self._transactions_df is not None else "non eseguita"
        return (
            f"AMLSim(num_accounts={self.config.num_accounts}, "
            f"num_steps={self.config.num_steps}, "
            f"seed={self.config.seed}, "
            f"status='{status}')"
        )
