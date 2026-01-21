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


def read_monthly_panel(
    project_id: str,
    client_key: str,
    months_back: int,
    ga_fallback_aov: float = 0.0,
) -> pd.DataFrame:
    """
    Devuelve panel mensual (TN con fallback GA) + costs.
    Intenta primero usar gold.ga_daily (si existe) y si falla cae a gold.ga_monthly.
    """
    bq = _bq_client(project_id)

    daily_sql = f"""
    -- GA DAILY → MONTHLY
    WITH ga_daily AS (
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
    tn AS (
      SELECT client_key, month,
             SUM(gross_revenue) AS purchase_revenue,
             SUM(orders)        AS purchases
      FROM `{project_id}.gold.tn_sales_monthly`
      WHERE client_key = @client_key
        AND month >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL @m_back MONTH), MONTH)
      GROUP BY 1,2
    ),
    meta AS (
      SELECT client_key, month, SUM(spend) AS cost_meta
      FROM `{project_id}.gold.ads_monthly`
      WHERE client_key = @client_key AND platform = 'meta-ads'
      GROUP BY 1,2
    ),
    months AS (
      SELECT client_key, month FROM tn
      UNION DISTINCT SELECT client_key, month FROM ga
      UNION DISTINCT SELECT client_key, month FROM meta
    )
    SELECT
      m.client_key,
      m.month,
      COALESCE(
        tn.purchase_revenue,
        NULLIF(ga.ga_revenue, 0),
        (NULLIF(ga.ga_purchases, 0) * NULLIF(ga.ga_aov, 0)),
        (NULLIF(ga.ga_purchases, 0) * NULLIF(@ga_aov, 0)),
        0.0
      ) AS purchase_revenue,
      COALESCE(tn.purchases, ga.ga_purchases, 0) AS purchases,
      COALESCE(ga.ucr, 0.0)                      AS user_conv_rate,
      COALESCE(meta.cost_meta, 0.0)              AS cost_meta,
      0.0                                        AS cost_google
    FROM months m
    LEFT JOIN tn   USING (client_key, month)
    LEFT JOIN ga   USING (client_key, month)
    LEFT JOIN meta USING (client_key, month)
    ORDER BY month
    """

    monthly_sql = f"""
    WITH tn AS (
      SELECT client_key, month,
             SUM(gross_revenue) AS purchase_revenue,
             SUM(orders)        AS purchases
      FROM `{project_id}.gold.tn_sales_monthly`
      WHERE client_key = @client_key
        AND month >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL @m_back MONTH), MONTH)
      GROUP BY 1,2
    ),
    meta AS (
      SELECT client_key, month, SUM(spend) AS cost_meta
      FROM `{project_id}.gold.ads_monthly`
      WHERE client_key = @client_key AND platform = 'meta-ads'
      GROUP BY 1,2
    ),
    ga AS (
      SELECT client_key, month,
             SAFE_DIVIDE(SUM(transactions), NULLIF(SUM(sessions),0)) AS ucr,
             SUM(transactions)    AS ga_purchases,
             SUM(purchaseRevenue) AS ga_revenue
      FROM `{project_id}.gold.ga_monthly`
      WHERE client_key = @client_key
      GROUP BY 1,2
    ),
    months AS (
      SELECT client_key, month FROM tn
      UNION DISTINCT SELECT client_key, month FROM ga
      UNION DISTINCT SELECT client_key, month FROM meta
    )
    SELECT
      m.client_key,
      m.month,
      COALESCE(
        tn.purchase_revenue,
        NULLIF(ga.ga_revenue, 0),
        (NULLIF(ga.ga_purchases, 0) * NULLIF(@ga_aov, 0)),
        0.0
      ) AS purchase_revenue,
      COALESCE(tn.purchases, ga.ga_purchases, 0) AS purchases,
      COALESCE(ga.ucr, 0.0)                      AS user_conv_rate,
      COALESCE(meta.cost_meta, 0.0)              AS cost_meta,
      0.0                                        AS cost_google
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
    """
    Devuelve (followers, engagement_rate) del último registro IG en gold.social_metrics.
    """
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
    """
    Lee el último registro de gold.tiktok_merchant_metrics (30d).
    """
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
    """
    v1: usa ER, comment_rate, share_rate, views_per_follower, consistency.
    Normalizaciones simples (caps) para 0..1 estable.
    Devuelve (score01, is_observable_30d).
    """
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


def blend_rrss(
    ig01: Optional[float],
    tt01: Optional[float],
    policy: str = "split",
) -> Optional[float]:
    if ig01 is None and tt01 is None:
        return None

    if policy == "best":
        return max(x for x in [ig01, tt01] if x is not None)

    if policy == "tt_only":
        return tt01

    if policy == "ig_only":
        return ig01

    # split
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
    full_df["ROAS"] = full_df.apply(
        lambda r: r["Purchase revenue"] / r["Cost_total"] if r["Cost_total"] > 0 else 0, axis=1
    )

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

    full_df["puntaje_conversion"] = full_df["User conversion rate"].apply(
        lambda x: puntuar_con_benchmark(x, benchmark_conversion, min_conversion)
    )
    full_df["puntaje_roas"] = full_df["ROAS"].apply(lambda x: puntuar_con_benchmark(x, benchmark_roas, min_roas))
    full_df["puntaje_ventas"] = full_df["Purchase revenue"].apply(
        lambda x: puntuar_con_benchmark(x, benchmark_ventas, min_ventas)
    )
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


# ── Overlays opcionales (merchant / bcra) ─────────────────────────────────────
def compute_merchant_basic(merchant_json: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any], List[str]]:
    try:
        products = merchant_json.get("metrics", {}).get("products", {}).get("products", []) or []
    except Exception:
        products = []
    n = len(products)
    if n == 0:
        return None, {"reason": "no_products"}, ["mc_missing"]

    ok_count = 0
    inv_vals: List[float] = []
    in_stock_count = 0

    for p in products:
        title = str(p.get("title") or "").strip()
        link = str(p.get("link") or "").strip()
        img = str(p.get("imageLink") or "").strip()
        avail = str(p.get("availability") or "").strip().lower()
        source = str(p.get("source") or "").strip().lower()

        has_min = bool(title and link and img)
        has_price = True
        if source == "feed":
            price = p.get("price") or {}
            has_price = bool(price.get("value")) and bool(price.get("currency"))

        avail_ok = avail in {"in stock", "preorder", "backorder"}
        status_ok = has_min and has_price and avail_ok
        ok_count += 1 if status_ok else 0

        if avail == "in stock":
            inv_vals.append(1.0)
            in_stock_count += 1
        elif avail in {"preorder", "backorder"}:
            inv_vals.append(0.5)
        else:
            inv_vals.append(0.0)

    product_status_ok_share = ok_count / n
    inventory_score = sum(inv_vals) / n if n > 0 else 0.0
    in_stock_share = in_stock_count / n if n > 0 else 0.0
    merchant_basic = 0.70 * product_status_ok_share + 0.30 * inventory_score

    flags: List[str] = []
    if product_status_ok_share < 0.70:
        flags.append("mc_low_status")
    if in_stock_share < 0.60:
        flags.append("mc_low_stock")

    debug = {
        "n_products": n,
        "product_status_ok_share": round(product_status_ok_share, 4),
        "inventory_score": round(inventory_score, 4),
        "in_stock_share": round(in_stock_share, 4),
        "merchant_basic": round(merchant_basic, 4),
    }
    return float(merchant_basic), debug, flags


def _months_back_from_ym(latest_ym: str, months: int = 12) -> Set[str]:
    y = int(latest_ym[:4])
    m = int(latest_ym[4:6])
    out = set()
    for i in range(months):
        yy = y
        mm = m - i
        while mm <= 0:
            yy -= 1
            mm += 12
        out.add(f"{yy:04d}{mm:02d}")
    return out


def bcra_extract_12m_from_historicas(bcra_json: Dict[str, Any]) -> Dict[str, Any]:
    hist = (bcra_json or {}).get("historicas", {}) or {}
    periods = hist.get("periodos", []) or []
    if not periods:
        return {"worst_situation_12m": None, "months_bad_12m": None, "entity_count_12m": 0, "periods_considered": []}

    latest_ym = max(str(p.get("periodo")) for p in periods if p.get("periodo"))
    win = _months_back_from_ym(latest_ym, months=12)

    per_period: Dict[str, List[Dict[str, Any]]] = {}
    for p in periods:
        ym = str(p.get("periodo"))
        if ym in win:
            per_period.setdefault(ym, []).extend(p.get("entidades", []) or [])

    worst = None
    months_bad = 0
    entities: Set[str] = set()

    for ym, entidades in per_period.items():
        worst_month = 0
        bad = False
        for e in entidades:
            sit = int(e.get("situacion", 0) or 0)
            ent_name = str(e.get("entidad") or "")
            if sit >= 1:
                entities.add(ent_name)
                worst_month = max(worst_month, sit)
                if sit >= 3:
                    bad = True
        if worst_month > 0:
            worst = worst_month if worst is None else max(worst, worst_month)
        if bad:
            months_bad += 1

    if worst is None:
        worst = 1

    return {
        "as_of_month": latest_ym,
        "worst_situation_12m": int(worst),
        "months_bad_12m": int(months_bad or 0),
        "entity_count_12m": len(entities),
        "periods_considered": sorted(per_period.keys()),
    }


def bcra_to_score_agg(
    worst_situation_12m: Optional[int],
    months_bad_12m: Optional[int],
    base_map: Dict[int, float] = None,
    months_coef: float = 0.08,
) -> Tuple[Optional[float], Dict[str, Any], List[str]]:
    if worst_situation_12m is None:
        return None, {"reason": "missing_worst"}, ["bcra_missing"]

    if base_map is None:
        base_map = {1: 1.00, 2: 0.85, 3: 0.55, 4: 0.25, 5: 0.00, 6: 0.00}

    base = base_map.get(int(worst_situation_12m), 0.0)
    months_penalty = 1.0
    if months_bad_12m is not None:
        months_penalty = max(0.5, 1.0 - months_coef * int(months_bad_12m))

    score = max(0.0, min(1.0, base * months_penalty))

    flags: List[str] = []
    if worst_situation_12m >= 5:
        flags.append("bcra_critical")
    elif worst_situation_12m == 4 and (months_bad_12m is not None and months_bad_12m >= 3):
        flags.append("bcra_high_risk")

    dbg = {
        "worst_situation_12m": int(worst_situation_12m),
        "months_bad_12m": int(months_bad_12m or 0),
        "base": base,
        "months_penalty": months_penalty,
        "score_0_1": score,
    }
    return float(score), dbg, flags


def blend_overlay(base01: float, merchant01: Optional[float], bcra01: Optional[float], wM: float = 0.08, wB: float = 0.07) -> float:
    parts: List[float] = [base01]
    weights: List[float] = [max(0.0, 1.0 - wM - wB)]
    if merchant01 is not None:
        parts.append(merchant01)
        weights.append(wM)
    if bcra01 is not None:
        parts.append(bcra01)
        weights.append(wB)
    s = sum(weights)
    if s <= 0:
        return base01
    weights = [w / s for w in weights]
    return sum(p * w for p, w in zip(parts, weights))


def apply_bcra_overrides(final01: float, bcra_flags: List[str]) -> float:
    if "bcra_critical" in bcra_flags:
        return min(final01, 0.35)
    if "bcra_high_risk" in bcra_flags:
        return min(final01, 0.55)
    return final01


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


# ── Entry function (ex-Prefect flow) ─────────────────────────────────────────
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
) -> int:
    """
    Calcula score final (1..1000), lo persiste en target_table y devuelve el entero.
    """

    ensure_target_table(project_id, target_table)

    # 1) Panel mensual (TN con fallback GA)
    panel = read_monthly_panel(project_id, client_key, months_back, ga_fallback_aov=ga_fallback_aov)
    if panel.empty:
        raise RuntimeError("No hay datos de panel mensual para el rango solicitado.")

    # 2) IG: si no vinieron overrides, leo de BQ
    ig_seguidores: Optional[int] = seguidores
    ig_engagement: Optional[float] = engagement_rate_redes
    if ig_seguidores is None or ig_engagement is None:
        try:
            ig_seguidores, ig_engagement = read_latest_ig_metrics(project_id, client_key)
            logger.info(f"[IG] métricas (BQ): seguidores={ig_seguidores}, engagement={ig_engagement}")
        except Exception as e:
            logger.warning(f"[IG] no disponible: {e}. Política RRSS='{rrss_policy}'.")
            ig_seguidores, ig_engagement = None, None
    else:
        logger.info(f"[IG] overrides recibidos: seguidores={ig_seguidores}, engagement={ig_engagement}")

    # 2.1) Scoring base (RRSS=IG provisorio)
    scored = apply_monthly_scoring(
        panel,
        seguidores=ig_seguidores,
        engagement_rate_redes=ig_engagement,
        rrss_policy=rrss_policy,
    )

    # 2.2) TikTok features → score 0..1
    tiktok_row = None
    try:
        tiktok_row = read_latest_tiktok_features(project_id, client_key)
    except Exception as e:
        logger.warning(f"[TIKTOK] no disponible: {e}")

    tt01, tt_obs = compute_tiktok_score01(tiktok_row)
    tt01 = tt01 if (tt01 is not None and tt_obs) else None

    # 2.3) IG componente 0..1
    ig01 = float(np.clip(scored["RRSS_score"].mean(), 0.0, 1.0)) if "RRSS_score" in scored.columns else None
    if ig_seguidores is None or ig_engagement is None:
        if rrss_policy == "renormalize":
            ig01 = None

    # 2.4) Blend IG+TikTok dentro del 15% RRSS
    rrss_component01 = blend_rrss(ig01, tt01, policy=rrss_blend_policy)
    logger.info(f"[RRSS] policy={rrss_blend_policy} ig01={ig01} tt01={tt01} -> rrss01={rrss_component01}")

    # 2.5) Recalcular score mensual con RRSS blended (peso 15%)
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

    # 3) Overlays opcionales
    m_score01: Optional[float] = None
    m_dbg: Dict[str, Any] = {}
    m_flags: List[str] = []
    if merchant_json is not None:
        m_score01, m_dbg, m_flags = compute_merchant_basic(merchant_json)

    b_score01: Optional[float] = None
    b_dbg: Dict[str, Any] = {}
    b_flags: List[str] = []
    if bcra_json is not None:
        agg = bcra_extract_12m_from_historicas(bcra_json)
        b_score01, b_dbg, b_flags = bcra_to_score_agg(
            worst_situation_12m=agg.get("worst_situation_12m"),
            months_bad_12m=agg.get("months_bad_12m"),
        )

    # 4) Blend + caps BCRA
    blended01 = blend_overlay(base01, m_score01, b_score01, wM=wM, wB=wB)
    blended01_capped = apply_bcra_overrides(blended01, b_flags)

    # 5) Escala final a 1..1000 entero
    score_value = int(round(blended01_capped * 1000.0))
    score_value = max(1, min(1000, score_value))

    # 6) Persistencia
    write_minimal_score(project_id, target_table, client_key, score_value)

    logger.info(
        f"[SCORE] client={client_key} "
        f"base100={base_score_value:.2f} rrss01={rrss_final01} "
        f"merchant={m_dbg} bcra={b_dbg} "
        f"final={score_value}"
    )

    return score_value
