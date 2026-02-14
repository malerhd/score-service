# scoring.py
from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Credenciales (tolerante) ─────────────────────────────────────────────────
def _get_gcp_credentials():
    raw = (os.getenv("SERVICE_ACCOUNT_B64") or "").strip()
    if not raw:
        return None  # ADC
    if (raw.startswith(("'", '"', "`")) and raw.endswith(raw[0])):
        raw = raw[1:-1].strip()
    try:
        if raw.lstrip().startswith("{"):
            info = json.loads(raw)
        else:
            s = re.sub(r"\s+", "", raw)
            s += "=" * (-len(s) % 4)
            info = json.loads(base64.b64decode(s).decode("utf-8"))
        return service_account.Credentials.from_service_account_info(info)
    except Exception as e:
        raise RuntimeError(f"SERVICE_ACCOUNT_B64 inválido (JSON/Base64): {e}")


def _bq_client(project_id: str) -> bigquery.Client:
    creds = _get_gcp_credentials()
    return bigquery.Client(project=project_id, credentials=creds) if creds else bigquery.Client(project=project_id)


# ── Normalización de fuentes activas ─────────────────────────────────────────
def _normalize_synced_sources(sources: Optional[List[str]]) -> Set[str]:
    """
    Normaliza fuentes activas recibidas del webhook/backend.
    Devuelve set en lower-case, sin espacios extra.
    """
    if not sources:
        return set()
    out = set()
    for s in sources:
        if s is None:
            continue
        out.add(str(s).strip().lower())
    return out


def _is_active(src_set: Set[str], candidates: Set[str]) -> bool:
    return any(c in src_set for c in candidates)


# ── IO BigQuery ──────────────────────────────────────────────────────────────
def ensure_target_table(project_id: str, target_table: str):
    bq = _bq_client(project_id)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{target_table}` (
      client_key STRING,
      score INT64
    )
    """
    bq.query(ddl).result()


def write_minimal_score(project_id: str, target_table: str, client_key: str, score: int):
    bq = _bq_client(project_id)

    bq.query(
        f"DELETE FROM `{target_table}` WHERE client_key = @client_key",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("client_key", "STRING", client_key)]
        ),
    ).result()

    out = pd.DataFrame([{"client_key": client_key, "score": int(score)}])
    bq.load_table_from_dataframe(out, target_table).result()


# ── Panel mensual con gating TN/GA (y opcional Meta) ─────────────────────────
def read_monthly_panel(
    project_id: str,
    client_key: str,
    months_back: int,
    ga_fallback_aov: float = 0.0,
    use_tn: bool = True,
    use_ga: bool = True,
    use_meta: bool = True,
) -> pd.DataFrame:
    """
    Devuelve panel mensual basado SOLO en fuentes activas:
      - use_tn: usa tn_sales_monthly
      - use_ga: usa ga_daily (si existe) sino ga_monthly
      - use_meta: usa ads_monthly(meta-ads) para costs
    """
    bq = _bq_client(project_id)

    # ✅ FIX: BigQuery no permite WHERE sin FROM.
    # Para CTE "vacío" usamos: FROM (SELECT 1) WHERE 1=0

    tn_cte = f"""
    tn AS (
      SELECT client_key, month,
             SUM(gross_revenue) AS purchase_revenue,
             SUM(orders)        AS purchases
      FROM `{project_id}.gold.tn_sales_monthly`
      WHERE client_key = @client_key
        AND month >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL @m_back MONTH), MONTH)
      GROUP BY 1,2
    ),
    """ if use_tn else """
    tn AS (
      SELECT
        CAST(NULL AS STRING)  AS client_key,
        CAST(NULL AS DATE)    AS month,
        CAST(0.0  AS FLOAT64) AS purchase_revenue,
        CAST(0    AS INT64)   AS purchases
      FROM (SELECT 1)
      WHERE 1=0
    ),
    """

    meta_cte = f"""
    meta AS (
      SELECT client_key, month, SUM(spend) AS cost_meta
      FROM `{project_id}.gold.ads_monthly`
      WHERE client_key = @client_key AND platform = 'meta-ads'
      GROUP BY 1,2
    ),
    """ if use_meta else """
    meta AS (
      SELECT
        CAST(NULL AS STRING)  AS client_key,
        CAST(NULL AS DATE)    AS month,
        CAST(0.0  AS FLOAT64) AS cost_meta
      FROM (SELECT 1)
      WHERE 1=0
    ),
    """

    ga_daily_cte = f"""
    ga_daily AS (
      SELECT
        client_key,
        DATE_TRUNC(PARSE_DATE('%Y%m%d', CAST(date AS STRING)), MONTH) AS month,
        SUM(sessions)        AS sessions,
        SUM(transactions)    AS transactions,
        SUM(purchaseRevenue) AS ga_revenue
      FROM `{project_id}.gold.ga_daily`
      WHERE client_key = @client_key
        AND PARSE_DATE('%Y%m%d', CAST(date AS STRING)) >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL @m_back MONTH), MONTH)
      GROUP BY 1,2
    ),
    ga AS (
      SELECT
        client_key, month,
        SAFE_DIVIDE(transactions, NULLIF(sessions,0)) AS ucr,
        transactions AS ga_purchases,
        ga_revenue,
        SAFE_DIVIDE(ga_revenue, NULLIF(transactions,0)) AS ga_aov
      FROM ga_daily
    ),
    """ if use_ga else """
    ga_daily AS (
      SELECT
        CAST(NULL AS STRING)  AS client_key,
        CAST(NULL AS DATE)    AS month,
        CAST(0    AS INT64)   AS sessions,
        CAST(0    AS INT64)   AS transactions,
        CAST(0.0  AS FLOAT64) AS ga_revenue
      FROM (SELECT 1)
      WHERE 1=0
    ),
    ga AS (
      SELECT
        CAST(NULL AS STRING)  AS client_key,
        CAST(NULL AS DATE)    AS month,
        CAST(0.0  AS FLOAT64) AS ucr,
        CAST(0    AS INT64)   AS ga_purchases,
        CAST(0.0  AS FLOAT64) AS ga_revenue,
        CAST(0.0  AS FLOAT64) AS ga_aov
      FROM (SELECT 1)
      WHERE 1=0
    ),
    """

    ga_monthly_cte = f"""
    ga AS (
      SELECT client_key, month,
             SAFE_DIVIDE(SUM(transactions), NULLIF(SUM(sessions),0)) AS ucr,
             SUM(transactions)    AS ga_purchases,
             SUM(purchaseRevenue) AS ga_revenue
      FROM `{project_id}.gold.ga_monthly`
      WHERE client_key = @client_key
      GROUP BY 1,2
    ),
    """ if use_ga else """
    ga AS (
      SELECT
        CAST(NULL AS STRING)  AS client_key,
        CAST(NULL AS DATE)    AS month,
        CAST(0.0  AS FLOAT64) AS ucr,
        CAST(0    AS INT64)   AS ga_purchases,
        CAST(0.0  AS FLOAT64) AS ga_revenue
      FROM (SELECT 1)
      WHERE 1=0
    ),
    """

    # Months union: solo fuentes activas
    months_union_parts = []
    if use_tn:
        months_union_parts.append("SELECT client_key, month FROM tn")
    if use_ga:
        months_union_parts.append("SELECT client_key, month FROM ga")
    if use_meta:
        months_union_parts.append("SELECT client_key, month FROM meta")

    if not months_union_parts:
        return pd.DataFrame()

    union_sql = "\n      UNION DISTINCT ".join(months_union_parts)

    months_cte = f"""
    months AS (
      {union_sql}
    )
    """

    revenue_expr = """
    COALESCE(
      tn.purchase_revenue,
      NULLIF(ga.ga_revenue, 0),
      (NULLIF(ga.ga_purchases, 0) * NULLIF(ga.ga_aov, 0)),
      (NULLIF(ga.ga_purchases, 0) * NULLIF(@ga_aov, 0)),
      0.0
    )
    """ if use_tn and use_ga else (
        "COALESCE(tn.purchase_revenue, 0.0)" if use_tn and not use_ga else
        "COALESCE(NULLIF(ga.ga_revenue, 0), (NULLIF(ga.ga_purchases, 0) * NULLIF(@ga_aov, 0)), 0.0)" if (not use_tn and use_ga) else
        "0.0"
    )

    purchases_expr = """
    COALESCE(tn.purchases, ga.ga_purchases, 0)
    """ if use_tn and use_ga else (
        "COALESCE(tn.purchases, 0)" if use_tn and not use_ga else
        "COALESCE(ga.ga_purchases, 0)" if (not use_tn and use_ga) else
        "0"
    )

    daily_sql = f"""
    WITH
    {ga_daily_cte}
    {tn_cte}
    {meta_cte}
    {months_cte}
    SELECT
      m.client_key,
      m.month,
      {revenue_expr} AS purchase_revenue,
      {purchases_expr} AS purchases,
      COALESCE(ga.ucr, 0.0)         AS user_conv_rate,
      COALESCE(meta.cost_meta, 0.0) AS cost_meta,
      0.0                           AS cost_google
    FROM months m
    LEFT JOIN tn   USING (client_key, month)
    LEFT JOIN ga   USING (client_key, month)
    LEFT JOIN meta USING (client_key, month)
    ORDER BY month
    """

    monthly_sql = f"""
    WITH
    {ga_monthly_cte}
    {tn_cte}
    {meta_cte}
    {months_cte}
    SELECT
      m.client_key,
      m.month,
      {revenue_expr} AS purchase_revenue,
      {purchases_expr} AS purchases,
      COALESCE(ga.ucr, 0.0)         AS user_conv_rate,
      COALESCE(meta.cost_meta, 0.0) AS cost_meta,
      0.0                           AS cost_google
    FROM months m
    LEFT JOIN tn   USING (client_key, month)
    LEFT JOIN ga   USING (client_key, month)
    LEFT JOIN meta USING (client_key, month)
    ORDER BY month
    """

    job_params = [
        bigquery.ScalarQueryParameter("client_key", "STRING", client_key),
        bigquery.ScalarQueryParameter("m_back", "INT64", months_back),
        bigquery.ScalarQueryParameter("ga_aov", "FLOAT64", ga_fallback_aov),
    ]
    cfg = bigquery.QueryJobConfig(query_parameters=job_params)

    try:
        df = bq.query(daily_sql, job_config=cfg).result().to_dataframe()
    except Exception as e:
        logger.warning(f"[BQ] ga_daily no disponible o falló (fallback a ga_monthly). Err: {e}")
        df = bq.query(monthly_sql, job_config=cfg).result().to_dataframe()

    df = df.rename(
        columns={
            "month": "Month",
            "purchase_revenue": "Purchase revenue",
            "purchases": "Purchases",
            "user_conv_rate": "User conversion rate",
            "cost_meta": "Cost_meta",
            "cost_google": "Cost_google",
        }
    )
    for c in ["Purchase revenue", "Purchases", "User conversion rate", "Cost_meta", "Cost_google"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def read_latest_ig_metrics(project_id: str, client_key: str) -> Tuple[int, float]:
    bq = _bq_client(project_id)
    sql = f"""
    SELECT followers, engagement_rate
    FROM `{project_id}.gold.social_metrics`
    WHERE client_key = @ck
      AND platform IN ('ig','instagram')
    ORDER BY fetched_at DESC
    LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ck", "STRING", client_key)])
    rows = list(bq.query(sql, job_config=cfg).result())
    if not rows:
        raise RuntimeError(f"[IG] No hay métricas en `{project_id}.gold.social_metrics` para client_key={client_key}.")
    followers = int(rows[0]["followers"] or 0)
    engagement = float(rows[0]["engagement_rate"] or 0.0)
    return followers, engagement


def read_latest_tiktok_features(project_id: str, client_key: str) -> Optional[dict]:
    bq = _bq_client(project_id)
    sql = f"""
    SELECT *
    FROM `{project_id}.gold.tiktok_merchant_metrics`
    WHERE client_key = @ck AND platform = 'tiktok'
    ORDER BY fetched_at DESC
    LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ck", "STRING", client_key)])
    rows = list(bq.query(sql, job_config=cfg).result())
    if not rows:
        return None
    return dict(rows[0].items())


# ── TikTok scoring 0–1 (v1) ─────────────────────────────────────────────────
def compute_tiktok_score01(row: Optional[dict]) -> Tuple[Optional[float], bool]:
    if not row:
        return None, False

    is_obs = bool(row.get("is_observable_30d", False))
    er = float(row.get("er_30d") or 0.0)
    cr = float(row.get("comment_rate_30d") or 0.0)
    sr = float(row.get("share_rate_30d") or 0.0)
    vpf = float(row.get("views_per_follower") or 0.0)
    cons = float(row.get("consistency_score") or 0.0)

    er01 = float(np.clip(er / 0.15, 0, 1))
    cr01 = float(np.clip(cr / 0.03, 0, 1))
    sr01 = float(np.clip(sr / 0.02, 0, 1))
    cons01 = float(np.clip(cons, 0, 1))
    vpf01 = float(np.clip(np.log1p(vpf) / np.log1p(50.0), 0, 1))

    quality = 0.6 * sr01 + 0.4 * cr01
    engagement = er01
    reach = 0.65 * vpf01 + 0.35 * sr01
    consistency = cons01
    growth = 0.0

    w = {"quality": 0.40, "eng": 0.25, "reach": 0.20, "growth": 0.10, "cons": 0.05}
    score01 = (
        w["quality"] * quality
        + w["eng"] * engagement
        + w["reach"] * reach
        + w["growth"] * growth
        + w["cons"] * consistency
    )

    if (vpf01 > 0.8) and (quality < 0.2) and (engagement < 0.2):
        score01 = min(score01, 0.35)

    return float(np.clip(score01, 0.0, 1.0)), is_obs


def blend_rrss(ig01: Optional[float], tt01: Optional[float], policy: str = "split") -> Optional[float]:
    if ig01 is None and tt01 is None:
        return None
    if policy == "best":
        return max(x for x in [ig01, tt01] if x is not None)
    if policy == "tt_only":
        return tt01
    if policy == "ig_only":
        return ig01
    if ig01 is not None and tt01 is not None:
        return 0.5 * ig01 + 0.5 * tt01
    return ig01 if ig01 is not None else tt01


# ── Base monthly scoring ─────────────────────────────────────────────────────
def apply_monthly_scoring(
    full_df: pd.DataFrame,
    seguidores: Optional[int] = None,
    engagement_rate_redes: Optional[float] = None,
    rrss_policy: str = "renormalize",
) -> pd.DataFrame:
    full_df["Cost_google"] = pd.to_numeric(full_df.get("Cost_google", 0), errors="coerce").fillna(0)
    full_df["Cost_meta"] = pd.to_numeric(full_df.get("Cost_meta", 0), errors="coerce").fillna(0)
    full_df["Purchase revenue"] = pd.to_numeric(full_df.get("Purchase revenue", 0), errors="coerce").fillna(0)
    full_df["Purchases"] = pd.to_numeric(full_df.get("Purchases", 0), errors="coerce").fillna(0)
    full_df["User conversion rate"] = pd.to_numeric(full_df.get("User conversion rate", 0), errors="coerce").fillna(0)

    full_df["Cost_total"] = full_df["Cost_google"] + full_df["Cost_meta"]
    full_df["CAC"] = full_df.apply(lambda r: r["Cost_total"] / r["Purchases"] if r["Purchases"] > 0 else 0, axis=1)
    full_df["ROAS"] = full_df.apply(lambda r: r["Purchase revenue"] / r["Cost_total"] if r["Cost_total"] > 0 else 0, axis=1)

    benchmark_conversion = 0.006
    benchmark_roas = 2.5
    benchmark_aov = 50000
    benchmark_cac_ideal = benchmark_aov * 0.10
    benchmark_cac_muy_alto = benchmark_aov * 0.25
    benchmark_ventas = 20_000_000
    min_conversion = 0.005
    min_roas = 1.5
    min_ventas = 5_000_000

    def puntuar_con_benchmark(v, b, minv=0.0):
        if v >= b:
            return 1.0
        if v <= minv:
            return 0.0
        return (v - minv) / (b - minv)

    def puntuar_cac(cac, ideal, maxv):
        if cac <= ideal:
            return 1.0
        if cac >= maxv:
            return 0.0
        return (maxv - cac) / (maxv - ideal)

    full_df["puntaje_conversion"] = full_df["User conversion rate"].apply(lambda x: puntuar_con_benchmark(x, benchmark_conversion, min_conversion))
    full_df["puntaje_roas"] = full_df["ROAS"].apply(lambda x: puntuar_con_benchmark(x, benchmark_roas, min_roas))
    full_df["puntaje_ventas"] = full_df["Purchase revenue"].apply(lambda x: puntuar_con_benchmark(x, benchmark_ventas, min_ventas))
    full_df["puntaje_cac"] = full_df["CAC"].apply(lambda x: puntuar_cac(x, benchmark_cac_ideal, benchmark_cac_muy_alto))

    rrss_disponible = (seguidores is not None) and (engagement_rate_redes is not None)
    if rrss_disponible:
        if engagement_rate_redes > 1:
            raise ValueError(f"[IG] engagement_rate {engagement_rate_redes} > 1 (usa proporción 0–1).")
        if seguidores < 0:
            raise ValueError(f"[IG] seguidores inválido: {seguidores}")
        engagement_rate_redes = float(np.clip(engagement_rate_redes, 0.0, 1.0))
        rrss_score = float(np.clip(np.log(seguidores + 1) * engagement_rate_redes, 0, 1))
    else:
        rrss_score = 0.5 if rrss_policy == "neutral" else 0.0

    full_df["RRSS_score"] = rrss_score
    full_df["puntaje_rrss"] = rrss_score

    w = {"ventas": 0.35, "roas": 0.20, "conv": 0.15, "cac": 0.15, "rrss": 0.15}
    if not rrss_disponible and rrss_policy == "renormalize":
        denom = w["ventas"] + w["roas"] + w["conv"] + w["cac"]  # 0.85
        w["ventas"] /= denom
        w["roas"] /= denom
        w["conv"] /= denom
        w["cac"] /= denom
        w["rrss"] = 0.0

    full_df["score_benchmark"] = (
        w["ventas"] * full_df["puntaje_ventas"]
        + w["roas"] * full_df["puntaje_roas"]
        + w["conv"] * full_df["puntaje_conversion"]
        + w["cac"] * full_df["puntaje_cac"]
        + w["rrss"] * full_df["puntaje_rrss"]
    )
    full_df["score_benchmark_100"] = full_df["score_benchmark"] * 100
    return full_df


# ── Entry function ───────────────────────────────────────────────────────────
def score_monthly_simple(
    project_id: str,
    client_key: str,
    target_table: str,
    months_back: int = 24,
    aggregate_last_n: int = 12,
    seguidores: Optional[int] = None,
    engagement_rate_redes: Optional[float] = None,
    rrss_policy: str = "renormalize",
    ga_fallback_aov: float = 0.0,
    merchant_json: Optional[Dict[str, Any]] = None,
    bcra_json: Optional[Dict[str, Any]] = None,
    wM: float = 0.08,
    wB: float = 0.07,
    rrss_blend_policy: str = "split",
    # ✅ NUEVO
    synced_sources: Optional[List[str]] = None,
) -> int:
    """
    Reglas pedidas:
      - Si TN activa → usa TN (si no hay datos, cae a GA si GA activa)
      - Si TN NO activa pero GA activa → usa GA
      - Si TN NO activa y GA NO activa → score=1
    """

    ensure_target_table(project_id, target_table)

    src = _normalize_synced_sources(synced_sources)

    # Definiciones de "activo" (nombres que pueden venir del backend)
    TN_KEYS = {"tienda nube", "tiendanube", "tn"}
    GA_KEYS = {"google analytics", "ga", "ga4", "analytics"}
    META_KEYS = {"meta ads", "meta", "meta-ads"}

    tn_active = _is_active(src, TN_KEYS) if src else True
    ga_active = _is_active(src, GA_KEYS) if src else True
    meta_active = _is_active(src, META_KEYS) if src else True

    logger.info(f"[ACTIVE] tn={tn_active} ga={ga_active} meta={meta_active} synced_sources={synced_sources}")

    # ✅ Si no hay TN ni GA activos → score=1
    if not tn_active and not ga_active:
        logger.warning("[SCORE] TN y GA inactivos → score=1 (no hay panel mensual).")
        write_minimal_score(project_id, target_table, client_key, 1)
        return 1

    # 1) Panel mensual (gated)
    panel = read_monthly_panel(
        project_id,
        client_key,
        months_back,
        ga_fallback_aov=ga_fallback_aov,
        use_tn=tn_active,
        use_ga=ga_active,
        use_meta=meta_active,
    )

    # Si TN estaba activa pero no hubo meses (o vino vacío) y GA está activa, reintento GA-only
    if panel.empty and ga_active:
        logger.warning("[SCORE] Panel vacío. Reintentando GA-only (TN ignored).")
        panel = read_monthly_panel(
            project_id,
            client_key,
            months_back,
            ga_fallback_aov=ga_fallback_aov,
            use_tn=False,
            use_ga=True,
            use_meta=meta_active,
        )

    if panel.empty:
        logger.warning("[SCORE] Panel mensual vacío incluso después de fallback → score=1.")
        write_minimal_score(project_id, target_table, client_key, 1)
        return 1

    # 2) IG / TikTok (igual que antes)
    ig_seguidores: Optional[int] = seguidores
    ig_engagement: Optional[float] = engagement_rate_redes
    if ig_seguidores is None or ig_engagement is None:
        try:
            ig_seguidores, ig_engagement = read_latest_ig_metrics(project_id, client_key)
            logger.info(f"[IG] métricas (BQ): seguidores={ig_seguidores}, engagement={ig_engagement}")
        except Exception as e:
            logger.warning(f"[IG] no disponible: {e}. Política RRSS='{rrss_policy}'.")
            ig_seguidores, ig_engagement = None, None

    scored = apply_monthly_scoring(
        panel,
        seguidores=ig_seguidores,
        engagement_rate_redes=ig_engagement,
        rrss_policy=rrss_policy,
    )

    tiktok_row = None
    try:
        tiktok_row = read_latest_tiktok_features(project_id, client_key)
    except Exception as e:
        logger.warning(f"[TIKTOK] no disponible: {e}")

    tt01, tt_obs = compute_tiktok_score01(tiktok_row)
    tt01 = tt01 if (tt01 is not None and tt_obs) else None

    ig01 = float(np.clip(scored["RRSS_score"].mean(), 0.0, 1.0)) if "RRSS_score" in scored.columns else None
    if ig_seguidores is None or ig_engagement is None:
        if rrss_policy == "renormalize":
            ig01 = None

    rrss_component01 = blend_rrss(ig01, tt01, policy=rrss_blend_policy)
    logger.info(f"[RRSS] policy={rrss_blend_policy} ig01={ig01} tt01={tt01} -> rrss01={rrss_component01}")

    w = {"ventas": 0.35, "roas": 0.20, "conv": 0.15, "cac": 0.15, "rrss": 0.15}
    rrss_final01 = rrss_component01 if rrss_component01 is not None else (0.5 if rrss_policy == "neutral" else 0.0)

    scored["rrss_component01_override"] = rrss_final01
    scored["score_benchmark_rrss_override"] = (
        w["ventas"] * scored["puntaje_ventas"]
        + w["roas"] * scored["puntaje_roas"]
        + w["conv"] * scored["puntaje_conversion"]
        + w["cac"] * scored["puntaje_cac"]
        + w["rrss"] * scored["rrss_component01_override"]
    )

    base_score_value = float(scored["score_benchmark_rrss_override"].tail(aggregate_last_n).mean() * 100.0)
    base01 = max(0.0, min(1.0, base_score_value / 100.0))

    # Overlays opcionales: no tocamos (en esta versión mínima quedan apagados si no pasan JSON)
    blended01 = base01
    score_value = int(round(blended01 * 1000.0))
    score_value = max(1, min(1000, score_value))

    write_minimal_score(project_id, target_table, client_key, score_value)

    logger.info(f"[SCORE] client={client_key} base100={base_score_value:.2f} final={score_value}")
    return score_value
