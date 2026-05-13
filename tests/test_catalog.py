from pathlib import Path

import matplotlib
import pandas as pd

from src.data.catalog import SatelliteCatalog
from src.evaluation.eda import (
    plot_metadata_missing_values,
    plot_ra_dec_distributions,
    plot_ra_dec_ranges,
    plot_satellite_name_frequency,
    plot_trail_length_distribution,
)


matplotlib.use("Agg")


def test_satellite_catalog_summary_methods(tmp_path: Path) -> None:
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "Length": [10.0, 20.0, 30.0],
            "Start_RA": [1.0, 2.0, 3.0],
            "End_RA": [1.5, 2.5, 3.5],
            "Start_DEC": [-1.0, 0.0, 1.0],
            "End_DEC": [-0.5, 0.5, 1.5],
            "Satellite_Name": ["A", "B", "A"],
        }
    ).to_csv(csv_path, index=False)

    catalog = SatelliteCatalog(csv_path)

    assert catalog.get_unique_satellites() == ["A", "B"]
    assert catalog.get_length_summary()["median"] == 20.0
    assert catalog.get_ra_dec_ranges() == {"ra": (1.0, 3.5), "dec": (-1.0, 1.5)}
    assert catalog.get_metadata_summary()["rows"] == 3


def test_catalog_plots_are_saved(tmp_path: Path) -> None:
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "Length": [10.0, 20.0, 30.0],
            "Start_RA": [1.0, 2.0, 3.0],
            "End_RA": [1.5, 2.5, 3.5],
            "Start_DEC": [-1.0, 0.0, 1.0],
            "End_DEC": [-0.5, 0.5, 1.5],
            "Satellite_Name": ["A", "B", "A"],
        }
    ).to_csv(csv_path, index=False)

    catalog = SatelliteCatalog(csv_path)

    frequency_path = plot_satellite_name_frequency(
        catalog,
        output_dir=tmp_path,
        show=False,
    )
    length_path = plot_trail_length_distribution(
        catalog,
        output_dir=tmp_path,
        show=False,
    )
    coordinates_path = plot_ra_dec_ranges(
        catalog,
        output_dir=tmp_path,
        show=False,
    )
    missing_path = plot_metadata_missing_values(
        catalog,
        output_dir=tmp_path,
        show=False,
    )
    distribution_path = plot_ra_dec_distributions(
        catalog,
        output_dir=tmp_path,
        show=False,
    )

    assert frequency_path.exists()
    assert length_path.exists()
    assert coordinates_path.exists()
    assert missing_path.exists()
    assert distribution_path.exists()
