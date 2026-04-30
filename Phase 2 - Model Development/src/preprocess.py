from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_RAW_PATH = Path(
    r"D:\Courses\DEPI R4 - Microsoft ML\Graduation Project\Phase 0 - About Dataset\Dataset\Metro.csv"
)
MILESTONE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = MILESTONE_ROOT / "data" / "processed" / "metro_clean.CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the MetroPT compressor dataset for Milestone 2 modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("METRO_CSV_PATH", DEFAULT_RAW_PATH)),
        help="Path to the raw Metro.csv file. Can also be set with METRO_CSV_PATH.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the cleaned CSV will be saved.",
    )
    parser.add_argument(
        "--resample-rule",
        default="10s",
        help="Pandas resampling rule for timestamp alignment.",
    )
    parser.add_argument(
        "--interpolation-limit",
        type=int,
        default=6,
        help="Maximum consecutive resampled gaps to interpolate.",
    )
    return parser.parse_args()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lower().replace(" ", "_").strip() for col in df.columns]
    return df


def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {input_path}")
    return pd.read_csv(input_path)


def clean_data(df: pd.DataFrame, resample_rule: str, interpolation_limit: int) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = normalize_column_names(df)
    if "timestamp" not in df.columns:
        raise ValueError("Missing required timestamp column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.resample(resample_rule).mean()
    df = df.interpolate(
        method="time",
        limit=interpolation_limit,
        limit_direction="both",
    )

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "tp2",
        "tp3",
        "reservoirs",
        "oil_temperature",
        "motor_current",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["pressure_delta"] = df["tp3"] - df["tp2"]
    df["power_indicator"] = df["motor_current"] * df["reservoirs"]
    df["oil_temp_rolling"] = df["oil_temperature"].rolling(window=60).mean()
    df["pressure_change"] = df["tp3"].diff()
    df["current_per_pressure"] = df["motor_current"] / (df["tp2"] + 0.1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def preprocess(
    input_path: Path,
    output_path: Path,
    resample_rule: str = "10s",
    interpolation_limit: int = 6,
) -> pd.DataFrame:
    raw_df = load_raw_data(input_path)
    clean_df = clean_data(raw_df, resample_rule, interpolation_limit)
    featured_df = add_features(clean_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_df.reset_index().to_csv(output_path, index=False)

    return featured_df


def main() -> None:
    args = parse_args()
    processed_df = preprocess(
        input_path=args.input,
        output_path=args.output,
        resample_rule=args.resample_rule,
        interpolation_limit=args.interpolation_limit,
    )

    print("Preprocessing completed.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(processed_df):,}")
    print(f"Columns: {len(processed_df.columns) + 1:,}")
    print(f"Start: {processed_df.index.min()}")
    print(f"End: {processed_df.index.max()}")


if __name__ == "__main__":
    main()
