"""Metadata access helpers for the satellite trail catalogue."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class SatelliteCatalog:
    """Load and query the satellite trail metadata catalogue."""

    REQUIRED_COLUMNS = {
        "Length",
        "Start_RA",
        "End_RA",
        "Start_DEC",
        "End_DEC",
        "Satellite_Name",
    }

    def __init__(self, csv_path: str | Path) -> None:
        """
        Load the catalogue from a CSV file.

        Parameters
        ----------
        csv_path:
            Path to `Satellites_Catalog_Application.csv`.
        """
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self._validate()

    def _validate(self) -> None:
        """Validate that the expected metadata columns are present."""
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def get_unique_satellites(self) -> list[str]:
        """Return the sorted list of unique satellite names."""
        return sorted(self.df["Satellite_Name"].dropna().unique().tolist())

    def get_metadata_summary(self) -> dict[str, int]:
        """Return headline counts for the catalogue."""
        return {
            "rows": int(len(self.df)),
            "unique_satellites": int(self.df["Satellite_Name"].nunique(dropna=True)),
            "labelled_rows": int(self.df["Satellite_Name"].notna().sum()),
            "missing_satellite_names": int(self.df["Satellite_Name"].isna().sum()),
        }

    def get_missing_value_counts(self) -> pd.Series:
        """Return missing-value counts for each metadata column."""
        return self.df.isna().sum()

    def get_satellite_frequency(self, top_n: int | None = None) -> pd.Series:
        """
        Count how often each satellite appears in the catalogue.

        Parameters
        ----------
        top_n:
            Optional limit on the number of entries returned.
        """
        frequency = self.df["Satellite_Name"].value_counts()
        return frequency.head(top_n) if top_n is not None else frequency

    def get_trail_lengths(self) -> pd.Series:
        """Return the trail length series as numeric values."""
        return pd.to_numeric(self.df["Length"], errors="coerce").dropna()

    def get_length_summary(self) -> dict[str, float]:
        """Summarise the trail length distribution."""
        lengths = self.get_trail_lengths()
        return {
            "count": float(lengths.count()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "min": float(lengths.min()),
            "max": float(lengths.max()),
        }

    def get_ra_dec_ranges(self) -> dict[str, tuple[float, float]]:
        """Return the overall right ascension and declination ranges."""
        ra_values = pd.concat([self.df["Start_RA"], self.df["End_RA"]])
        dec_values = pd.concat([self.df["Start_DEC"], self.df["End_DEC"]])
        return {
            "ra": (float(ra_values.min()), float(ra_values.max())),
            "dec": (float(dec_values.min()), float(dec_values.max())),
        }

    def get_coordinate_dataframe(self) -> pd.DataFrame:
        """Return the coordinate columns used for plotting and analysis."""
        columns = [
            "Satellite_Name",
            "Start_RA",
            "End_RA",
            "Start_DEC",
            "End_DEC",
            "Length",
        ]
        return self.df.loc[:, columns].copy()

    def get_numeric_summary(self) -> pd.DataFrame:
        """Return summary statistics for the numeric metadata columns."""
        columns = ["Length", "Start_RA", "End_RA", "Start_DEC", "End_DEC"]
        return self.df.loc[:, columns].describe().transpose()
