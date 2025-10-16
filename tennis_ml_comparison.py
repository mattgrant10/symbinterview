# =============================
# FILE: tennis_ml_comparison.py
# =============================

"""
TENNIS PERFORMANCE PREDICTION: Traditional ML vs Symbolic ML
=============================================================
This script demonstrates the fundamental differences between:
1. Black-box machine learning (Random Forest, XGBoost, Neural Networks)
2. Glass-box symbolic regression (Mathematical equation discovery)

Dataset: Tennis matches 2017-2023 (Excel files 2017.xlsx ... 2023.xlsx)
Goal: Predict match outcomes/player performance with interpretable models

Notes
-----
• All DataFrames are printed with `tabulate` for readability.
• Ultra-detailed logging is enabled and saved to `tennis_ml_education.log`.
• The code is defensive: it checks for missing columns and degrades gracefully.
• Symbolic ML requires PySR. If unavailable, we skip with a friendly message.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pathlib import Path

# Default folder for raw yearly Excel files
HERE = Path(__file__).parent
DEFAULT_DATA_DIR = HERE / "Data1_ATP_symbint"

import numpy as np
import pandas as pd
from tabulate import tabulate

# --------------------------------------------------------------------------------------
# Logging setup (ultra‑detailed educational style)
# --------------------------------------------------------------------------------------
LOG_FILE = "tennis_ml_education.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s | LEARNING: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("SYSTEM INFORMATION FOR REPRODUCIBILITY")
logger.info(f"Python Version: {sys.version}")
logger.info(f"Current Directory: {os.getcwd()}")
logger.info(f"Available CPU cores: {os.cpu_count()}")
logger.info("=" * 80)

# --------------------------------------------------------------------------------------
# Pretty printing helper
# --------------------------------------------------------------------------------------

MAX_COLS_PER_TABLE = 7  # hard cap on columns per printed table

def _chunk_columns(df: pd.DataFrame, max_cols: int = MAX_COLS_PER_TABLE):
    """Split DataFrame into chunks of max_cols columns each."""
    cols = list(df.columns)
    if not cols:
        yield df
        return
    for start in range(0, len(cols), max_cols):
        yield df.iloc[:, start:start + max_cols]

def print_df(df: pd.DataFrame, title: str, max_rows: int = 10) -> None:
    """Print a DataFrame with tabulate, splitting into <= MAX_COLS_PER_TABLE columns.

    Always called whenever a DataFrame is constructed or used for clarity.
    """
    if df is None:
        logger.warning(f"{title}: DataFrame is None")
        return

    total_cols = df.shape[1]
    logger.info("\n" + "-" * 80)
    logger.info(f"TABLE: {title} — total columns: {total_cols} (showing up to {max_rows} rows per slice)")
    logger.info("-" * 80)

    slice_idx = 1
    head = df.head(max_rows)
    for sub in _chunk_columns(head):
        if len(sub.columns) > 0:
            hdr = f"{title} — slice {slice_idx} (cols {sub.columns[0]} … {sub.columns[-1]})"
        else:
            hdr = f"{title} — slice {slice_idx}"
        table = tabulate(sub, headers="keys", tablefmt="github", showindex=False)
        logger.info("\n" + hdr + "\n" + table + "\n")
        print("\n" + hdr)
        print(table)
        slice_idx += 1

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

class TennisDataLoader:
    """
    TEACHING: This class demonstrates proper data engineering practices.
    We separate data loading logic for reusability and testing.
    """

    def __init__(
        self,
        years_range: Tuple[int, int] = (2017, 2023),
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ):
        self.years = range(years_range[0], years_range[1] + 1)
        self.data_dir = Path(data_dir)
        logger.info(
            f"Initializing data loader for years: {list(self.years)}"
            f" and data directory: {self.data_dir}"
        )

    def load_and_combine(self) -> pd.DataFrame:
        """
        LEARNING POINT: Data concatenation and preprocessing
        - Why: Multiple year files need to become one dataset
        - How: pandas concat stacks DataFrames vertically
        """
        all_data: List[pd.DataFrame] = []
        for year in self.years:
            filename = self.data_dir / f"{year}.xlsx"
            logger.info("\n" + "=" * 60)
            logger.info(f"LOADING: {filename}")
            try:
                df = pd.read_excel(filename, sheet_name=0)
                logger.info(f"✓ Loaded {len(df)} rows from {year}")
                logger.info(f"  Columns found (first 10): {list(df.columns)[:10]}")
                logger.info(f"  Data shape: {df.shape} (rows × columns)")
                logger.info(
                    f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB"
                )
                df["year"] = year
                all_data.append(df)
            except FileNotFoundError:
                logger.warning(f"✗ File {filename} not found - skipping")
            except Exception as e:
                logger.error(f"✗ Error loading {filename}: {str(e)}")

        if not all_data:
            raise FileNotFoundError(
                "No yearly Excel files were found (2017.xlsx ... 2023.xlsx)."
            )

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info("\n" + "=" * 60)
        logger.info("COMBINED DATASET STATISTICS:")
        logger.info(f"Total matches: {len(combined_df)}")
        logger.info(f"Date range: {combined_df['year'].min()} - {combined_df['year'].max()}")
        logger.info(f"Total features: {len(combined_df.columns)}")
        print_df(combined_df, title="Combined dataset (head)")
        return combined_df

# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------

TENNIS_FEATURES_EXPLANATIONS: Dict[str, str] = {
    # Service metrics
    'aces_per_match': 'Aggressive serving indicator',
    'double_faults_per_match': 'Service pressure indicator',
    'first_serve_percentage': 'Service consistency',
    'first_serve_points_won': 'Service effectiveness',
    # Return metrics
    'break_points_converted': 'Return game strength',
    'return_points_won': 'Defensive ability',
    # Overall metrics
    'total_points_won_percentage': 'Overall dominance',
    'tiebreaks_won_percentage': 'Clutch performance',
    # Physical/conditional
    'minutes_per_set': 'Fitness/efficiency indicator',
    'distance_covered': 'Court coverage ability',
    # Historical/derived
    'h2h_record': 'Psychological advantage',
    'recent_form': 'Last 5 matches win rate',
    'surface_specialist_score': 'Surface-specific performance',
}

class TennisFeatureEngineer:
    """
    TEACHING: Feature engineering is where domain knowledge meets ML
    Tennis-specific features that algorithms can learn from
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ENGINEERING: Creating meaningful tennis metrics")
        logger.info("=" * 80)

    def _safe_ratio(self, num: pd.Series, den: pd.Series) -> pd.Series:
        return num.astype(float) / np.where(den.astype(float) == 0, np.nan, den.astype(float))

    def create_features(self) -> pd.DataFrame:
        # SURFACE TYPE ENCODING
        logger.info("\n1. SURFACE TYPE ENCODING")
        if "surface" in self.df.columns:
            surface_encoded = pd.get_dummies(self.df["surface"], prefix="surface")
            self.df = pd.concat([self.df, surface_encoded], axis=1)
            surface_map = {"Hard": 1.0, "Clay": 0.5, "Grass": 0.75}
            self.df["surface_numeric"] = self.df["surface"].map(surface_map)
            logger.info(f"   One-hot columns: {list(surface_encoded.columns)}")
            logger.info(f"   Numeric mapping used: {surface_map}")
        else:
            logger.warning("   Column 'surface' not found — skipping surface encodings.")

        # PLAYER PERFORMANCE METRICS (defensive coding against missing cols)
        logger.info("\n2. PLAYER PERFORMANCE METRICS — rolling averages/ratios where available")
        # Example source columns (adjust if present):
        # 'Aces', 'DoubleFaults', 'SrvPnts', '1stIn', '1stWon', 'BPSaved', 'BPFaced', 'RetPntsWon'
        colmap = {
            'ace_rate': ("Aces", "SrvPnts"),
            'first_serve_success': ("1stIn", "SrvPnts"),
            'first_serve_points_won_rate': ("1stWon", "1stIn"),
            'break_point_saved_rate': ("BPSaved", "BPFaced"),
            'return_points_won_rate': ("RetPntsWon", "RetPntsPlayed"),
        }
        for feat, (num, den) in colmap.items():
            if num in self.df.columns and den in self.df.columns:
                self.df[feat] = self._safe_ratio(self.df[num], self.df[den])
                logger.info(f"   • {feat} = {num}/{den}")
            else:
                logger.warning(f"   • Skipping {feat}: missing columns {num} or {den}")

        # Rolling form example (by player if columns exist)
        player_col = None
        for c in ["player", "Player", "player_name", "name"]:
            if c in self.df.columns:
                player_col = c
                break
        if player_col is not None and "won_match" in self.df.columns:
            logger.info("   • recent_form: rolling win-rate over last 5 matches per player")
            self.df.sort_values([player_col, "year"], inplace=True)
            self.df["recent_form"] = (
                self.df.groupby(player_col)["won_match"].apply(lambda s: s.rolling(5, min_periods=1).mean())
            ).values
        else:
            logger.warning("   • Skipping recent_form: need 'won_match' and a player column")

        # Surface specialist dummy (toy example)
        if "surface" in self.df.columns and player_col is not None:
            logger.info("   • surface_specialist_score: per-player mean win-rate by surface")
            try:
                winrate_by_surface = (
                    self.df.groupby([player_col, "surface"]).apply(
                        lambda d: d["won_match"].mean() if "won_match" in d.columns else np.nan
                    )
                ).rename("surface_winrate").reset_index()
                self.df = self.df.merge(winrate_by_surface, on=[player_col, "surface"], how="left")
                self.df.rename(columns={"surface_winrate": "surface_specialist_score"}, inplace=True)
            except Exception as e:
                logger.warning(f"   • Could not compute surface_specialist_score: {e}")

        print_df(self.df, title="Feature‑engineered dataset (head)")
        return self.df

# --------------------------------------------------------------------------------------
# Modeling utilities
# --------------------------------------------------------------------------------------

TARGET_CANDIDATES = [
    # Binary outcome style
    "won_match", "Winner", "winner", "win",
    # Regression style (performance proxy)
    "total_points_won_percentage", "TPW%", "tpw_pct",
]

EXCLUDE_COLUMNS = {"date", "Date", "match_id", "id", "notes", "surface"}

@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    task: str  # "classification" or "regression"


def choose_target(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Select a sensible target. Prefer binary if available, else regression proxy."""

    # First, try to create a binary target from the Winner column
    # We'll predict whether the higher-ranked player won (1) or lost (0)
    if "Winner" in df.columns and "WRank" in df.columns and "LRank" in df.columns:
        # Create binary target: 1 if winner had better (lower) rank, 0 otherwise
        y = (df["WRank"] < df["LRank"]).astype(int)
        logger.info("Target created: higher_ranked_won (classification) - predicting if better-ranked player won")
        return y, "classification"

    # Try other predefined target candidates
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            y = df[col]
            # Binary if 2 unique values only
            uniq = pd.Series(y).dropna().unique()
            if len(uniq) == 2:
                logger.info(f"Target selected: {col} (classification)")
                return y.astype(int), "classification"
            else:
                # Only use as regression if it's already numeric
                y_numeric = pd.to_numeric(y, errors="coerce")
                if not y_numeric.isna().all():
                    logger.info(f"Target selected: {col} (regression)")
                    return y_numeric, "regression"

    # Fallback: create a proxy target if service/return columns exist
    proxy_cols = ["1stWon", "2ndWon", "RetPntsWon", "SrvPnts"]
    if all(c in df.columns for c in proxy_cols):
        y = (0.6 * df["1stWon"] + 0.4 * df["2ndWon"]) / df["SrvPnts"]
        logger.warning("No explicit target found — using service points efficiency proxy (regression)")
        return y, "regression"
    raise ValueError("No suitable target column found. Please include one of: " + ", ".join(TARGET_CANDIDATES))


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> PreparedData:
    from sklearn.model_selection import train_test_split

    y, task = choose_target(df)

    # Select numeric features; drop target and obvious identifiers
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if y.name in numeric_df.columns:
        numeric_df.drop(columns=[y.name], inplace=True)

    for col in list(EXCLUDE_COLUMNS):
        if col in numeric_df.columns:
            numeric_df.drop(columns=[col], inplace=True)

    # Drop columns with all-NaN or zero variance
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how="all")
    nunique = numeric_df.nunique()
    keep_cols = nunique[nunique > 1].index.tolist()
    X = numeric_df[keep_cols].copy()

    # Impute remaining NaNs simply (median)
    X = X.fillna(X.median(numeric_only=True))

    # Show the modeling frame
    print_df(X, title="Model features X (head)")
    logger.info(f"Model will use {X.shape[1]} numeric features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task == "classification" else None
    )

    print_df(pd.DataFrame({"y_train": y_train}).head(10), title="Target y_train (head)")
    print_df(pd.DataFrame({"y_test": y_test}).head(10), title="Target y_test (head)")

    return PreparedData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, task=task)

# --------------------------------------------------------------------------------------
# Traditional ML (Random Forest, XGBoost, MLP)
# --------------------------------------------------------------------------------------

class TraditionalMLPipeline:
    """
    TEACHING: Traditional ML - The Black Box Approach
    These models learn patterns but don't give us equations
    """

    def __init__(self, prepared: PreparedData):
        self.X_train = prepared.X_train
        self.X_test = prepared.X_test
        self.y_train = prepared.y_train
        self.y_test = prepared.y_test
        self.task = prepared.task

        logger.info("\n" + "=" * 80)
        logger.info("TRADITIONAL MACHINE LEARNING PIPELINE")
        logger.info("=" * 80)
        logger.info("Approach: Learn patterns from data without explicit formulas")
        logger.info(f"Training samples: {len(self.X_train)}")
        logger.info(f"Testing samples: {len(self.X_test)}")
        logger.info(f"Features: {self.X_train.shape[1]}")

    def _metrics(self, y_true, y_pred) -> Dict[str, float]:
        if self.task == "classification":
            from sklearn.metrics import accuracy_score, f1_score
            yhat = (np.array(y_pred) >= 0.5).astype(int) if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)
            return {"accuracy": float(accuracy_score(y_true, yhat)), "f1": float(f1_score(y_true, yhat))}
        else:
            from sklearn.metrics import r2_score, mean_squared_error
            return {"r2": float(r2_score(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}

    def train_random_forest(self) -> Tuple[object, Dict[str, float]]:
        logger.info("\n" + "-" * 60)
        logger.info("MODEL 1: RANDOM FOREST")
        logger.info("-" * 60)
        logger.info("CONCEPT: Like asking 100 tennis experts and averaging their opinions")

        if self.task == "classification":
            from sklearn.ensemble import RandomForestClassifier as RFC
            model = RFC(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor as RFR
            model = RFR(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)

        model.fit(self.X_train, self.y_train)
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        train_metrics = self._metrics(self.y_train, train_pred)
        test_metrics = self._metrics(self.y_test, test_pred)

        logger.info(f"RESULTS (train): {train_metrics}")
        logger.info(f"RESULTS (test):  {test_metrics}")

        # Feature importances (if available)
        if hasattr(model, "feature_importances_"):
            importances = (
                pd.DataFrame({"feature": self.X_train.columns, "importance": model.feature_importances_})
                .sort_values("importance", ascending=False)
            )
            print_df(importances.head(15), title="RandomForest — top feature importances")
        else:
            logger.info("Model has no feature_importances_ attribute.")

        return model, test_metrics

    def train_xgboost(self) -> Tuple[object, Dict[str, float]]:
        logger.info("\n" + "-" * 60)
        logger.info("MODEL 2: XGBoost (Gradient Boosting)")
        logger.info("-" * 60)
        try:
            import xgboost as xgb
        except Exception as e:
            logger.warning(f"XGBoost not available: {e} — skipping.")
            return None, {"note": "xgboost not installed"}

        if self.task == "classification":
            model = xgb.XGBClassifier(
                n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric="logloss"
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=42, n_jobs=-1
            )

        model.fit(self.X_train, self.y_train)
        test_pred = model.predict(self.X_test)
        test_metrics = self._metrics(self.y_test, test_pred)
        logger.info(f"RESULTS (test): {test_metrics}")
        return model, test_metrics

    def train_neural_network(self) -> Tuple[object, Dict[str, float]]:
        logger.info("\n" + "-" * 60)
        logger.info("MODEL 3: Neural Network (MLP)")
        logger.info("-" * 60)
        if self.task == "classification":
            from sklearn.neural_network import MLPClassifier as MLP
            model = MLP(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", max_iter=200, random_state=42)
        else:
            from sklearn.neural_network import MLPRegressor as MLP
            model = MLP(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", max_iter=300, random_state=42)

        model.fit(self.X_train, self.y_train)
        test_pred = model.predict(self.X_test)
        test_metrics = self._metrics(self.y_test, test_pred)
        logger.info(f"RESULTS (test): {test_metrics}")
        return model, test_metrics

# --------------------------------------------------------------------------------------
# Symbolic ML (PySR)
# --------------------------------------------------------------------------------------

class SymbolicMLPipeline:
    """
    TEACHING: Symbolic Regression - The Glass Box Approach
    Discovers actual mathematical equations from data
    """

    def __init__(self, prepared: PreparedData):
        self.X_train = prepared.X_train
        self.X_test = prepared.X_test
        self.y_train = prepared.y_train
        self.y_test = prepared.y_test
        self.task = prepared.task

        logger.info("\n" + "=" * 80)
        logger.info("SYMBOLIC MACHINE LEARNING PIPELINE")
        logger.info("=" * 80)
        logger.info("Approach: Discover mathematical equations that explain the data")
        logger.info("Output: Human-readable formulas like physics equations")

    def train_pysr(self) -> Tuple[Optional[object], Dict[str, float]]:
        """Run PySR and print equations clearly in the console as well as logs."""
        if self.task == "classification":
            logger.warning("PySR is regression-only; fitting to numeric target as regression.")

        try:
            from pysr import PySRRegressor
        except Exception as e:
            logger.warning(f"PySR not available: {e} — skipping symbolic regression.")
            print("\n[Symbolic ML] PySR not available — skipping symbolic regression.")
            return None, {"note": "pysr not installed"}

        from sklearn.metrics import r2_score, mean_squared_error

        logger.info("\n" + "-" * 60)
        logger.info("SYMBOLIC REGRESSION WITH PySR")
        logger.info("-" * 60)
        logger.info("Genetic programming steps: initialize → evaluate → select → mutate → crossover → repeat")

        # Note: 'verbose' parameter removed as it's not supported in this PySR version
        model = PySRRegressor(
            niterations=60,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt", "square"],
            population_size=120,
            maxsize=32,
            parsimony=0.001,
            random_state=42,
            procs=max(1, (os.cpu_count() or 2) // 2),
        )

        logger.info("Starting evolution…")
        model.fit(self.X_train.values, self.y_train.values)

        # 1) Show Pareto-front equations (chunked to ≤7 columns)
        try:
            eq_df = getattr(model, "equations_", None)
            if eq_df is not None and not eq_df.empty:
                # Select relevant columns if they exist
                keep = [c for c in ["equation", "loss", "complexity", "score", "r2"] if c in eq_df.columns]
                view = eq_df[keep] if keep else eq_df
                print_df(view.head(10), title="PySR Pareto-front equations (top)")
                # Save all equations to file
                with open("discovered_equations.txt", "w", encoding="utf-8") as f:
                    f.write(view.to_string(index=False))
            else:
                logger.info("No equations_ attribute exposed by PySR version.")
        except Exception as e:
            logger.warning(f"Could not log equations table: {e}")

        # 2) Metrics
        test_pred = model.predict(self.X_test.values)
        r2 = float(r2_score(self.y_test, test_pred))
        rmse = float(np.sqrt(mean_squared_error(self.y_test, test_pred)))
        metrics = {"r2": r2, "rmse": rmse}
        logger.info(f"FINAL PERFORMANCE (test): {metrics}")

        # 3) Print the single best equation prominently (chat/console)
        try:
            best_expr = model.sympy()
            banner = "=" * 60
            msg = f"\n{banner}\nBEST SYMBOLIC EQUATION (interpretable formula)\n{banner}\n{best_expr}\n{banner}\n"
            print(msg)
            logger.info(msg)
        except Exception as e:
            msg = f"\n[Symbolic ML] Could not render best equation via sympy(): {e}"
            print(msg)
            logger.info("sympy() not available on this model/version.")

        return model, metrics

# --------------------------------------------------------------------------------------
# Comparison helper
# --------------------------------------------------------------------------------------

class ModelComparison:
    """TEACHING: Direct comparison of approaches"""

    def __init__(self, traditional_results: Dict[str, float], symbolic_results: Dict[str, float]):
        self.traditional = traditional_results
        self.symbolic = symbolic_results

    def compare_and_visualize(self) -> pd.DataFrame:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL COMPARISON: TRADITIONAL ML vs SYMBOLIC ML")
        logger.info("=" * 80)

        # Normalize keys
        t_r2 = self.traditional.get("r2") or self.traditional.get("accuracy") or np.nan
        s_r2 = self.symbolic.get("r2", np.nan)

        comparison = {
            "Metric": [
                "Accuracy (R²/Acc)",
                "Interpretability",
                "Equation Output",
                "Extrapolation Ability",
                "Feature Engineering Need",
                "Domain Insight",
            ],
            "Traditional ML": [
                f"{t_r2:.4f}" if isinstance(t_r2, float) else str(t_r2),
                "Low (black box)",
                "No",
                "Often limited",
                "High",
                "Limited",
            ],
            "Symbolic ML": [
                f"{s_r2:.4f}" if isinstance(s_r2, float) else str(s_r2),
                "High (equation)",
                "Yes — explicit formula",
                "Often better (model is analytic)",
                "Medium",
                "High",
            ],
        }
        df_comparison = pd.DataFrame(comparison)
        print_df(df_comparison, title="Traditional vs Symbolic — comparison table")

        logger.info("\nKEY INSIGHTS:\n1) Traditional: better raw predictive horsepower with enough data.\n"
                    "2) Symbolic: reveals mechanism-level relationships (Occam’s razor).\n"
                    "3) Hybrid: use symbolic to craft features, then feed to traditional models.")
        return df_comparison

# --------------------------------------------------------------------------------------
# Orchestration helper (can also be used programmatically; accepts years range and data directory)
# --------------------------------------------------------------------------------------

def run_full_pipeline(
    years: Tuple[int, int] = (2017, 2023),
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Dict[str, object]:
    loader = TennisDataLoader(years_range=years, data_dir=data_dir)
    df_raw = loader.load_and_combine()

    engineer = TennisFeatureEngineer(df_raw)
    df_feat = engineer.create_features()

    prepared = prepare_train_test(df_feat)

    # Traditional models
    trad = TraditionalMLPipeline(prepared)
    rf_model, rf_metrics = trad.train_random_forest()
    xgb_model, xgb_metrics = trad.train_xgboost()
    mlp_model, mlp_metrics = trad.train_neural_network()

    # Choose the best traditional metric to summarize (r2 if regression else accuracy)
    def pick_metric(m: Dict[str, float]) -> float:
        for k in ("r2", "accuracy"):
            v = m.get(k)
            if v is not None and isinstance(v, (int, float)):
                return float(v)
        return np.nan

    best_trad_score = np.nanmax([pick_metric(rf_metrics), pick_metric(xgb_metrics), pick_metric(mlp_metrics)])

    # Symbolic regression
    sym = SymbolicMLPipeline(prepared)
    sr_model, sr_metrics = sym.train_pysr()

    comparison = ModelComparison(
        traditional_results={"r2": best_trad_score},
        symbolic_results={"r2": sr_metrics.get("r2", np.nan)},
    )
    comparison_df = comparison.compare_and_visualize()

    return {
        "df_raw": df_raw,
        "df_feat": df_feat,
        "prepared": prepared,
        "rf": (rf_model, rf_metrics),
        "xgb": (xgb_model, xgb_metrics),
        "mlp": (mlp_model, mlp_metrics),
        "sr": (sr_model, sr_metrics),
        "comparison": comparison_df,
    }


# =============================
# FILE: run_tennis_analysis.py
# =============================

"""
MAIN EXECUTION SCRIPT
This orchestrates the entire comparison with educational output
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tennis data analysis: traditional ML vs symbolic regression"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Directory containing yearly Excel data files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--start-year", type=int, default=2017,
        help="First year to include (inclusive)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2023,
        help="Last year to include (inclusive)"
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet", action="store_true",
        help="Suppress INFO-level console logging"
    )
    verbosity.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging"
    )
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.debug:
        logger.setLevel(logging.DEBUG)

    print("\n" + "=" * 80)
    print("TENNIS DATA ANALYSIS: LEARNING TRADITIONAL vs SYMBOLIC ML")
    print("=" * 80)
    print(
        f"\nLoading data from {args.data_dir}, years {args.start_year}-{args.end_year}\n"
    )

    try:
        results = run_full_pipeline(
            years=(args.start_year, args.end_year),
            data_dir=args.data_dir,
        )
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE! Check tennis_ml_education.log and discovered_equations.txt")
    print("=" * 80)
