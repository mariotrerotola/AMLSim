#!/usr/bin/env python3
"""Advanced feature engineering for AML detection.

Builds ~80+ account-level features from AMLSim outputs:
- Original structural/monetary/temporal/cash features (baseline)
- Graph/network features (PageRank, centrality, community)
- Temporal windowed features (7/30/90-day windows, entropy)
- Transaction pattern features (round amounts, Gini, Herfindahl)
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

EPS = 1e-9


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_conf(conf_path: Path) -> dict:
    with conf_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_sim_name(conf: dict, override: str | None = None) -> str:
    if override:
        return override
    return conf.get("general", {}).get("simulation_name", "sample")


def resolve_output_dir(conf: dict, sim_name: str) -> Path:
    base = conf.get("output", {}).get("directory", "outputs")
    return Path(base) / sim_name


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _mean_interval_seconds(ts: pd.Series) -> float:
    ts_sorted = ts.dropna().sort_values()
    if len(ts_sorted) <= 1:
        return 0.0
    diffs = ts_sorted.diff().dt.total_seconds().dropna()
    return float(diffs.mean()) if len(diffs) else 0.0


def _std_interval_seconds(ts: pd.Series) -> float:
    ts_sorted = ts.dropna().sort_values()
    if len(ts_sorted) <= 1:
        return 0.0
    diffs = ts_sorted.diff().dt.total_seconds().dropna()
    return float(diffs.std()) if len(diffs) else 0.0


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution (normalised counts)."""
    p = counts / (counts.sum() + EPS)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + EPS)))


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of an array of values."""
    if len(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_v) - (n + 1) * np.sum(sorted_v)) / (n * np.sum(sorted_v) + EPS))


def _herfindahl_index(counts: np.ndarray) -> float:
    """Herfindahl-Hirschman index (market concentration)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    shares = counts / total
    return float(np.sum(shares ** 2))


# ---------------------------------------------------------------------------
# 1. BASELINE FEATURES (from original script, refactored)
# ---------------------------------------------------------------------------

def _build_baseline_features(
    accounts: pd.DataFrame,
    tx: pd.DataFrame,
    cash_tx: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the original 34 baseline features."""

    incoming = tx[["bene_acct", "orig_acct", "base_amt", "tran_timestamp"]].rename(
        columns={"bene_acct": "acct_id", "orig_acct": "counterparty",
                 "base_amt": "in_amount", "tran_timestamp": "in_ts"}
    )
    outgoing = tx[["orig_acct", "bene_acct", "base_amt", "tran_timestamp"]].rename(
        columns={"orig_acct": "acct_id", "bene_acct": "counterparty",
                 "base_amt": "out_amount", "tran_timestamp": "out_ts"}
    )

    in_stats = incoming.groupby("acct_id").agg(
        n_in_tx=("counterparty", "count"),
        total_in_amt=("in_amount", "sum"),
        avg_in_amt=("in_amount", "mean"),
        std_in_amt=("in_amount", "std"),
        unique_in_ctp=("counterparty", "nunique"),
        first_in_ts=("in_ts", "min"),
        last_in_ts=("in_ts", "max"),
    )
    out_stats = outgoing.groupby("acct_id").agg(
        n_out_tx=("counterparty", "count"),
        total_out_amt=("out_amount", "sum"),
        avg_out_amt=("out_amount", "mean"),
        std_out_amt=("out_amount", "std"),
        unique_out_ctp=("counterparty", "nunique"),
        first_out_ts=("out_ts", "min"),
        last_out_ts=("out_ts", "max"),
    )

    in_intervals = incoming.groupby("acct_id")["in_ts"].apply(_mean_interval_seconds).rename("avg_in_interval_sec")
    out_intervals = outgoing.groupby("acct_id")["out_ts"].apply(_mean_interval_seconds).rename("avg_out_interval_sec")
    in_interval_std = incoming.groupby("acct_id")["in_ts"].apply(_std_interval_seconds).rename("std_in_interval_sec")
    out_interval_std = outgoing.groupby("acct_id")["out_ts"].apply(_std_interval_seconds).rename("std_out_interval_sec")

    # Cash stats
    cash_in = cash_tx[cash_tx["tx_type"] == "CASH-IN"].copy()
    cash_out = cash_tx[cash_tx["tx_type"] == "CASH-OUT"].copy()
    cash_in["acct_id"] = cash_in["orig_acct"].astype(str)
    cash_out["acct_id"] = cash_out["bene_acct"].astype(str)

    cash_in_stats = cash_in.groupby("acct_id").agg(
        cash_in_count=("base_amt", "count"),
        cash_in_total=("base_amt", "sum"),
        cash_in_avg=("base_amt", "mean"),
    )
    cash_out_stats = cash_out.groupby("acct_id").agg(
        cash_out_count=("base_amt", "count"),
        cash_out_total=("base_amt", "sum"),
        cash_out_avg=("base_amt", "mean"),
    )

    base = accounts[["acct_id", "initial_deposit", "bank_id"]].copy()
    for col in ["acct_id", "bank_id"]:
        base[col] = base[col].astype(str)

    features = base.set_index("acct_id")
    for df in [in_stats, out_stats, in_intervals, out_intervals,
               in_interval_std, out_interval_std, cash_in_stats, cash_out_stats]:
        features = features.join(df, how="left")

    # Lifetime
    first_ts = pd.concat([features["first_in_ts"], features["first_out_ts"]], axis=1).min(axis=1)
    last_ts = pd.concat([features["last_in_ts"], features["last_out_ts"]], axis=1).max(axis=1)
    lifetime_sec = (last_ts - first_ts).dt.total_seconds().fillna(0.0)

    # Fill NaN
    fill_cols = [
        "n_in_tx", "n_out_tx", "total_in_amt", "total_out_amt",
        "avg_in_amt", "avg_out_amt", "std_in_amt", "std_out_amt",
        "unique_in_ctp", "unique_out_ctp",
        "avg_in_interval_sec", "avg_out_interval_sec",
        "std_in_interval_sec", "std_out_interval_sec",
        "cash_in_count", "cash_out_count",
        "cash_in_total", "cash_out_total",
        "cash_in_avg", "cash_out_avg",
    ]
    for c in fill_cols:
        features[c] = features[c].fillna(0.0)

    features["n_total_tx"] = features["n_in_tx"] + features["n_out_tx"]
    features["unique_in_ratio"] = features["unique_in_ctp"] / (features["n_in_tx"] + EPS)
    features["unique_out_ratio"] = features["unique_out_ctp"] / (features["n_out_tx"] + EPS)
    features["in_out_ratio"] = features["n_in_tx"] / (features["n_out_tx"] + EPS)
    features["volume_ratio"] = features["total_out_amt"] / (features["total_in_amt"] + EPS)
    features["net_balance_ratio"] = (
        (features["total_in_amt"] - features["total_out_amt"])
        / (features["total_in_amt"] + features["total_out_amt"] + EPS)
    )
    features["weighted_avg_tx"] = (
        (features["avg_in_amt"] * features["n_in_tx"] + features["avg_out_amt"] * features["n_out_tx"])
        / (features["n_total_tx"] + EPS)
    )
    features["wallet_lifetime_sec"] = lifetime_sec
    features["activity_index"] = features["n_total_tx"] / (lifetime_sec / 86400.0 + 1.0)
    features["time_interval_ratio"] = features["avg_out_interval_sec"] / (features["avg_in_interval_sec"] + EPS)
    features["cash_flow_ratio"] = features["cash_out_total"] / (features["cash_in_total"] + EPS)
    features["burstiness_in"] = features["std_in_interval_sec"] / (features["avg_in_interval_sec"] + EPS)
    features["burstiness_out"] = features["std_out_interval_sec"] / (features["avg_out_interval_sec"] + EPS)

    # Drop non-numeric columns used for computation
    features = features.drop(columns=["first_in_ts", "last_in_ts", "first_out_ts", "last_out_ts", "bank_id"])

    return features


# ---------------------------------------------------------------------------
# 2. GRAPH / NETWORK FEATURES
# ---------------------------------------------------------------------------

def _build_graph_features(tx: pd.DataFrame, all_acct_ids: pd.Index) -> pd.DataFrame:
    """Build graph-theoretic features from the transaction network."""

    # Build weighted directed graph
    edge_weights = tx.groupby(["orig_acct", "bene_acct"]).agg(
        weight=("base_amt", "sum"),
        count=("base_amt", "count"),
    ).reset_index()

    G = nx.DiGraph()
    G.add_nodes_from(all_acct_ids)
    for _, row in edge_weights.iterrows():
        G.add_edge(row["orig_acct"], row["bene_acct"],
                    weight=row["weight"], count=row["count"])

    # PageRank
    try:
        pr = nx.pagerank(G, weight="weight", max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G, weight="weight", max_iter=300, tol=1e-4)
    pr_series = pd.Series(pr, name="pagerank")

    # Degree centrality
    in_deg_c = pd.Series(nx.in_degree_centrality(G), name="in_degree_centrality")
    out_deg_c = pd.Series(nx.out_degree_centrality(G), name="out_degree_centrality")

    # Undirected graph for clustering and community
    G_undir = G.to_undirected()

    clustering = pd.Series(nx.clustering(G_undir), name="clustering_coeff")

    # Betweenness centrality (approximate for performance)
    n_nodes = G.number_of_nodes()
    k_samples = min(500, n_nodes)
    bc = nx.betweenness_centrality(G, k=k_samples, weight="weight", seed=42)
    bc_series = pd.Series(bc, name="betweenness_centrality")

    # K-core number
    kcore = nx.core_number(G_undir)
    kcore_series = pd.Series(kcore, name="kcore_number")

    # Community detection (Louvain)
    try:
        communities = nx.community.louvain_communities(G_undir, seed=42)
        node_community = {}
        community_sizes = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                node_community[node] = idx
            community_sizes[idx] = len(comm)
        comm_id = pd.Series(node_community, name="community_id")
        comm_size = pd.Series(
            {n: community_sizes[c] for n, c in node_community.items()},
            name="community_size",
        )
    except Exception:
        comm_id = pd.Series(0, index=all_acct_ids, name="community_id")
        comm_size = pd.Series(n_nodes, index=all_acct_ids, name="community_size")

    # Average neighbor degree
    avg_neigh_in = {}
    avg_neigh_out = {}
    for node in G.nodes():
        preds = list(G.predecessors(node))
        succs = list(G.successors(node))
        avg_neigh_in[node] = np.mean([G.in_degree(p) for p in preds]) if preds else 0.0
        avg_neigh_out[node] = np.mean([G.out_degree(s) for s in succs]) if succs else 0.0
    avg_neigh_in_s = pd.Series(avg_neigh_in, name="avg_neighbor_in_degree")
    avg_neigh_out_s = pd.Series(avg_neigh_out, name="avg_neighbor_out_degree")

    # Reciprocity per node: fraction of counterparties that are both sender and receiver
    reciprocity = {}
    for node in G.nodes():
        preds = set(G.predecessors(node))
        succs = set(G.successors(node))
        union = preds | succs
        if len(union) == 0:
            reciprocity[node] = 0.0
        else:
            reciprocity[node] = len(preds & succs) / len(union)
    reciprocity_s = pd.Series(reciprocity, name="reciprocity")

    # 2-hop reach (number of unique nodes within 2 hops)
    two_hop_reach = {}
    for node in G.nodes():
        one_hop = set(G.successors(node)) | set(G.predecessors(node))
        two_hop = set()
        for n1 in one_hop:
            two_hop |= set(G.successors(n1)) | set(G.predecessors(n1))
        two_hop.discard(node)
        two_hop_reach[node] = len(two_hop)
    two_hop_s = pd.Series(two_hop_reach, name="two_hop_reach")

    graph_df = pd.DataFrame({
        "pagerank": pr_series,
        "in_degree_centrality": in_deg_c,
        "out_degree_centrality": out_deg_c,
        "clustering_coeff": clustering,
        "betweenness_centrality": bc_series,
        "kcore_number": kcore_series,
        "community_id": comm_id,
        "community_size": comm_size,
        "avg_neighbor_in_degree": avg_neigh_in_s,
        "avg_neighbor_out_degree": avg_neigh_out_s,
        "reciprocity": reciprocity_s,
        "two_hop_reach": two_hop_s,
    })

    return graph_df.reindex(all_acct_ids).fillna(0.0)


# ---------------------------------------------------------------------------
# 3. TEMPORAL WINDOWED FEATURES
# ---------------------------------------------------------------------------

def _build_temporal_features(tx: pd.DataFrame, all_acct_ids: pd.Index) -> pd.DataFrame:
    """Temporal windowed features: 7/30/90-day windows, entropy, day-of-week."""

    ts_min = tx["tran_timestamp"].min()
    ts_max = tx["tran_timestamp"].max()
    total_span = (ts_max - ts_min).days + 1

    results = {}

    for window_days in [7, 30, 90]:
        suffix = f"_{window_days}d"
        cutoff = ts_max - pd.Timedelta(days=window_days)
        recent = tx[tx["tran_timestamp"] >= cutoff]

        # Per-account counts and volumes in recent window
        out_recent = recent.groupby("orig_acct").agg(
            **{f"out_count{suffix}": ("base_amt", "count"),
               f"out_vol{suffix}": ("base_amt", "sum")}
        )
        in_recent = recent.groupby("bene_acct").agg(
            **{f"in_count{suffix}": ("base_amt", "count"),
               f"in_vol{suffix}": ("base_amt", "sum")}
        )
        out_recent.index.name = "acct_id"
        in_recent.index.name = "acct_id"

        for col in out_recent.columns:
            results[col] = out_recent[col]
        for col in in_recent.columns:
            results[col] = in_recent[col]

    # Early vs late ratio (trend detection)
    mid_point = ts_min + (ts_max - ts_min) / 2
    early = tx[tx["tran_timestamp"] < mid_point]
    late = tx[tx["tran_timestamp"] >= mid_point]

    early_out = early.groupby("orig_acct")["base_amt"].sum().rename("early_out_vol")
    late_out = late.groupby("orig_acct")["base_amt"].sum().rename("late_out_vol")
    early_in = early.groupby("bene_acct")["base_amt"].sum().rename("early_in_vol")
    late_in = late.groupby("bene_acct")["base_amt"].sum().rename("late_in_vol")
    early_out.index.name = "acct_id"
    late_out.index.name = "acct_id"
    early_in.index.name = "acct_id"
    late_in.index.name = "acct_id"

    results["early_out_vol"] = early_out
    results["late_out_vol"] = late_out
    results["early_in_vol"] = early_in
    results["late_in_vol"] = late_in

    # Day-of-week entropy (uniform = high entropy, concentrated = low)
    tx_copy = tx.copy()
    tx_copy["dow"] = tx_copy["tran_timestamp"].dt.dayofweek

    def _dow_entropy(group):
        dow_counts = np.zeros(7)
        for d in group:
            dow_counts[d] += 1
        return _shannon_entropy(dow_counts)

    dow_ent_out = tx_copy.groupby("orig_acct")["dow"].apply(_dow_entropy).rename("dow_entropy_out")
    dow_ent_in = tx_copy.groupby("bene_acct")["dow"].apply(_dow_entropy).rename("dow_entropy_in")
    dow_ent_out.index.name = "acct_id"
    dow_ent_in.index.name = "acct_id"
    results["dow_entropy_out"] = dow_ent_out
    results["dow_entropy_in"] = dow_ent_in

    # Temporal concentration: entropy of weekly bins
    if total_span > 7:
        tx_copy["week_bin"] = ((tx_copy["tran_timestamp"] - ts_min).dt.days // 7).astype(int)
        n_bins = tx_copy["week_bin"].max() + 1

        def _temporal_entropy(group):
            bin_counts = np.zeros(n_bins)
            for b in group:
                bin_counts[b] += 1
            return _shannon_entropy(bin_counts)

        temp_ent_out = tx_copy.groupby("orig_acct")["week_bin"].apply(_temporal_entropy).rename("temporal_entropy_out")
        temp_ent_in = tx_copy.groupby("bene_acct")["week_bin"].apply(_temporal_entropy).rename("temporal_entropy_in")
        temp_ent_out.index.name = "acct_id"
        temp_ent_in.index.name = "acct_id"
        results["temporal_entropy_out"] = temp_ent_out
        results["temporal_entropy_in"] = temp_ent_in

    temporal_df = pd.DataFrame(results)
    temporal_df = temporal_df.reindex(all_acct_ids).fillna(0.0)

    # Derived ratios
    temporal_df["trend_out_ratio"] = temporal_df["late_out_vol"] / (temporal_df["early_out_vol"] + EPS)
    temporal_df["trend_in_ratio"] = temporal_df["late_in_vol"] / (temporal_df["early_in_vol"] + EPS)

    return temporal_df


# ---------------------------------------------------------------------------
# 4. TRANSACTION PATTERN FEATURES
# ---------------------------------------------------------------------------

def _build_pattern_features(tx: pd.DataFrame, all_acct_ids: pd.Index) -> pd.DataFrame:
    """Transaction pattern features: round amounts, Gini, Herfindahl, etc."""

    results = {}

    # --- Outgoing patterns ---
    out_groups = tx.groupby("orig_acct")

    # Round amount ratios
    def _round_ratio(amounts, divisor):
        if len(amounts) == 0:
            return 0.0
        return float(np.sum(np.mod(amounts, divisor) == 0) / len(amounts))

    round_100_out = out_groups["base_amt"].apply(lambda x: _round_ratio(x.values, 100)).rename("round_100_ratio_out")
    round_1000_out = out_groups["base_amt"].apply(lambda x: _round_ratio(x.values, 1000)).rename("round_1000_ratio_out")
    round_100_out.index.name = "acct_id"
    round_1000_out.index.name = "acct_id"
    results["round_100_ratio_out"] = round_100_out
    results["round_1000_ratio_out"] = round_1000_out

    # Max / avg ratio (outlier proxy)
    max_avg_out = out_groups["base_amt"].apply(
        lambda x: float(x.max() / (x.mean() + EPS)) if len(x) > 0 else 0.0
    ).rename("max_avg_ratio_out")
    max_avg_out.index.name = "acct_id"
    results["max_avg_ratio_out"] = max_avg_out

    # Gini coefficient on outgoing amounts
    gini_out = out_groups["base_amt"].apply(
        lambda x: _gini_coefficient(x.values)
    ).rename("gini_out")
    gini_out.index.name = "acct_id"
    results["gini_out"] = gini_out

    # Herfindahl index on counterparty distribution
    herf_out = out_groups["bene_acct"].apply(
        lambda x: _herfindahl_index(x.value_counts().values)
    ).rename("herfindahl_out")
    herf_out.index.name = "acct_id"
    results["herfindahl_out"] = herf_out

    # --- Incoming patterns ---
    in_groups = tx.groupby("bene_acct")

    round_100_in = in_groups["base_amt"].apply(lambda x: _round_ratio(x.values, 100)).rename("round_100_ratio_in")
    round_1000_in = in_groups["base_amt"].apply(lambda x: _round_ratio(x.values, 1000)).rename("round_1000_ratio_in")
    round_100_in.index.name = "acct_id"
    round_1000_in.index.name = "acct_id"
    results["round_100_ratio_in"] = round_100_in
    results["round_1000_ratio_in"] = round_1000_in

    max_avg_in = in_groups["base_amt"].apply(
        lambda x: float(x.max() / (x.mean() + EPS)) if len(x) > 0 else 0.0
    ).rename("max_avg_ratio_in")
    max_avg_in.index.name = "acct_id"
    results["max_avg_ratio_in"] = max_avg_in

    gini_in = in_groups["base_amt"].apply(
        lambda x: _gini_coefficient(x.values)
    ).rename("gini_in")
    gini_in.index.name = "acct_id"
    results["gini_in"] = gini_in

    herf_in = in_groups["orig_acct"].apply(
        lambda x: _herfindahl_index(x.value_counts().values)
    ).rename("herfindahl_in")
    herf_in.index.name = "acct_id"
    results["herfindahl_in"] = herf_in

    # Amount coefficient of variation
    cv_out = out_groups["base_amt"].apply(
        lambda x: float(x.std() / (x.mean() + EPS)) if len(x) > 1 else 0.0
    ).rename("amount_cv_out")
    cv_in = in_groups["base_amt"].apply(
        lambda x: float(x.std() / (x.mean() + EPS)) if len(x) > 1 else 0.0
    ).rename("amount_cv_in")
    cv_out.index.name = "acct_id"
    cv_in.index.name = "acct_id"
    results["amount_cv_out"] = cv_out
    results["amount_cv_in"] = cv_in

    # Median amount (robust central tendency)
    med_out = out_groups["base_amt"].median().rename("median_out_amt")
    med_in = in_groups["base_amt"].median().rename("median_in_amt")
    med_out.index.name = "acct_id"
    med_in.index.name = "acct_id"
    results["median_out_amt"] = med_out
    results["median_in_amt"] = med_in

    # Skewness of amounts
    skew_out = out_groups["base_amt"].apply(
        lambda x: float(scipy_stats.skew(x.values)) if len(x) > 2 else 0.0
    ).rename("skewness_out")
    skew_in = in_groups["base_amt"].apply(
        lambda x: float(scipy_stats.skew(x.values)) if len(x) > 2 else 0.0
    ).rename("skewness_in")
    skew_out.index.name = "acct_id"
    skew_in.index.name = "acct_id"
    results["skewness_out"] = skew_out
    results["skewness_in"] = skew_in

    # Transaction type diversity (entropy over tx_type)
    def _tx_type_entropy(group):
        counts = group.value_counts().values
        return _shannon_entropy(counts)

    tx_type_ent_out = out_groups["tx_type"].apply(_tx_type_entropy).rename("tx_type_entropy_out")
    tx_type_ent_in = in_groups["tx_type"].apply(_tx_type_entropy).rename("tx_type_entropy_in")
    tx_type_ent_out.index.name = "acct_id"
    tx_type_ent_in.index.name = "acct_id"
    results["tx_type_entropy_out"] = tx_type_ent_out
    results["tx_type_entropy_in"] = tx_type_ent_in

    pattern_df = pd.DataFrame(results)
    return pattern_df.reindex(all_acct_ids).fillna(0.0)


# ---------------------------------------------------------------------------
# 5. MAIN BUILDER
# ---------------------------------------------------------------------------

def build_all_features(output_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full feature matrix and labels from AMLSim output directory.

    Returns (features_df, y_series) where features_df index is acct_id (str).
    """

    accounts_path = output_dir / "accounts.csv"
    tx_path = output_dir / "transactions.csv"
    cash_path = output_dir / "cash_tx.csv"
    sar_path = output_dir / "sar_accounts.csv"

    for p in [accounts_path, tx_path, cash_path, sar_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    accounts = pd.read_csv(accounts_path)
    tx = pd.read_csv(tx_path)
    cash_tx = pd.read_csv(cash_path)
    sar = pd.read_csv(sar_path)

    # Preprocess
    tx = tx.copy()
    tx["tran_timestamp"] = _to_datetime(tx["tran_timestamp"])
    tx["base_amt"] = pd.to_numeric(tx["base_amt"], errors="coerce").fillna(0.0)
    tx["orig_acct"] = tx["orig_acct"].astype(str)
    tx["bene_acct"] = tx["bene_acct"].astype(str)
    tx = tx[(tx["orig_acct"] != "-") & (tx["bene_acct"] != "-")]

    cash_tx = cash_tx.copy()
    cash_tx["base_amt"] = pd.to_numeric(cash_tx["base_amt"], errors="coerce").fillna(0.0)

    print("[features] Building baseline features...")
    baseline = _build_baseline_features(accounts, tx, cash_tx)
    all_acct_ids = baseline.index

    print("[features] Building graph features...")
    graph = _build_graph_features(tx, all_acct_ids)

    print("[features] Building temporal features...")
    temporal = _build_temporal_features(tx, all_acct_ids)

    print("[features] Building pattern features...")
    pattern = _build_pattern_features(tx, all_acct_ids)

    # Merge all
    features = baseline.join(graph, how="left").join(temporal, how="left").join(pattern, how="left")
    features = features.fillna(0.0)

    # Labels
    sar_accounts = sar["ACCOUNT_ID"].astype(str).unique()
    y = pd.Series(features.index.isin(sar_accounts).astype(int), index=features.index, name="is_sar")

    print(f"[features] Total features: {features.shape[1]} | Samples: {len(features)} | SAR rate: {y.mean():.4f}")

    return features, y


def build_baseline_features_only(output_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Build only the original 34 baseline features (for backward compatibility)."""

    accounts = pd.read_csv(output_dir / "accounts.csv")
    tx = pd.read_csv(output_dir / "transactions.csv")
    cash_tx = pd.read_csv(output_dir / "cash_tx.csv")
    sar = pd.read_csv(output_dir / "sar_accounts.csv")

    tx = tx.copy()
    tx["tran_timestamp"] = _to_datetime(tx["tran_timestamp"])
    tx["base_amt"] = pd.to_numeric(tx["base_amt"], errors="coerce").fillna(0.0)
    tx["orig_acct"] = tx["orig_acct"].astype(str)
    tx["bene_acct"] = tx["bene_acct"].astype(str)
    tx = tx[(tx["orig_acct"] != "-") & (tx["bene_acct"] != "-")]

    cash_tx = cash_tx.copy()
    cash_tx["base_amt"] = pd.to_numeric(cash_tx["base_amt"], errors="coerce").fillna(0.0)

    features = _build_baseline_features(accounts, tx, cash_tx)
    features = features.fillna(0.0)

    sar_accounts = sar["ACCOUNT_ID"].astype(str).unique()
    y = pd.Series(features.index.isin(sar_accounts).astype(int), index=features.index, name="is_sar")

    return features, y
