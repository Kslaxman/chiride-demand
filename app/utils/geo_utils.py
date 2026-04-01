"""
Geospatial utilities — compatible with h3 v3 AND v4.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Optional, Dict

try:
    import h3
    HAS_H3 = True

    if hasattr(h3, "latlng_to_cell"):
        def _latlng_to_h3(lat, lon, res):
            return h3.latlng_to_cell(lat, lon, res)
        def _h3_to_latlng(h3_index):
            return h3.cell_to_latlng(h3_index)
        def _k_ring(h3_index, k):
            return h3.grid_disk(h3_index, k)
    else:
        def _latlng_to_h3(lat, lon, res):
            return h3.geo_to_h3(lat, lon, res)
        def _h3_to_latlng(h3_index):
            return h3.h3_to_geo(h3_index)
        def _k_ring(h3_index, k):
            return h3.k_ring(h3_index, k)
except ImportError:
    HAS_H3 = False
    def _latlng_to_h3(lat, lon, res): return ""
    def _h3_to_latlng(h3_index): return (0, 0)
    def _k_ring(h3_index, k): return {h3_index}


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Chicago's 77 Community Areas with center coordinates

CHICAGO_COMMUNITY_AREAS = {
    "Albany Park": (41.9684, -87.7244),
    "Archer Heights": (41.8095, -87.7244),
    "Armour Square": (41.8422, -87.6324),
    "Ashburn": (41.7496, -87.7112),
    "Auburn Gresham": (41.7428, -87.6530),
    "Austin": (41.8933, -87.7651),
    "Avalon Park": (41.7458, -87.5887),
    "Avondale": (41.9389, -87.7112),
    "Belmont Cragin": (41.9314, -87.7651),
    "Beverly": (41.7159, -87.6733),
    "Bridgeport": (41.8382, -87.6500),
    "Brighton Park": (41.8192, -87.6950),
    "Burnside": (41.7290, -87.5981),
    "Calumet Heights": (41.7288, -87.5790),
    "Chatham": (41.7408, -87.6130),
    "Chicago Lawn": (41.7728, -87.6950),
    "Clearing": (41.7810, -87.7651),
    "Douglas": (41.8352, -87.6178),
    "Dunning": (41.9489, -87.8189),
    "East Garfield Park": (41.8802, -87.7042),
    "East Side": (41.7068, -87.5509),
    "Edgewater": (41.9833, -87.6603),
    "Edison Park": (42.0087, -87.8134),
    "Englewood": (41.7796, -87.6448),
    "Forest Glen": (41.9833, -87.7651),
    "Fuller Park": (41.8092, -87.6324),
    "Gage Park": (41.7948, -87.6950),
    "Garfield Ridge": (41.7948, -87.7651),
    "Grand Boulevard": (41.8122, -87.6178),
    "Greater Grand Crossing": (41.7608, -87.6130),
    "Hegewisch": (41.6528, -87.5509),
    "Hermosa": (41.9214, -87.7244),
    "Humboldt Park": (41.9024, -87.7244),
    "Hyde Park": (41.8022, -87.5887),
    "Irving Park": (41.9539, -87.7244),
    "Jefferson Park": (41.9704, -87.7651),
    "Kenwood": (41.8092, -87.5981),
    "Lake View": (41.9434, -87.6530),
    "Lincoln Park": (41.9214, -87.6500),
    "Lincoln Square": (41.9684, -87.6883),
    "Logan Square": (41.9234, -87.6883),
    "Loop": (41.8819, -87.6278),
    "Lower West Side": (41.8542, -87.6650),
    "McKinley Park": (41.8312, -87.6733),
    "Montclare": (41.9284, -87.8000),
    "Morgan Park": (41.6898, -87.6733),
    "Mount Greenwood": (41.6968, -87.7112),
    "Near North Side": (41.9002, -87.6324),
    "Near South Side": (41.8562, -87.6278),
    "Near West Side": (41.8802, -87.6650),
    "New City": (41.8092, -87.6583),
    "North Center": (41.9539, -87.6733),
    "North Lawndale": (41.8602, -87.7112),
    "North Park": (41.9833, -87.7244),
    "Norwood Park": (41.9860, -87.8134),
    "Oakland": (41.8222, -87.6037),
    "Ohare": (41.9833, -87.8500),
    "Portage Park": (41.9584, -87.7651),
    "Pullman": (41.6968, -87.6130),
    "Riverdale": (41.6388, -87.6324),
    "Rogers Park": (42.0087, -87.6733),
    "Roseland": (41.7068, -87.6278),
    "South Chicago": (41.7358, -87.5509),
    "South Deering": (41.7068, -87.5674),
    "South Lawndale": (41.8422, -87.7112),
    "South Shore": (41.7608, -87.5674),
    "Uptown": (41.9654, -87.6530),
    "Washington Heights": (41.7228, -87.6500),
    "Washington Park": (41.7928, -87.6178),
    "West Elsdon": (41.7948, -87.7244),
    "West Englewood": (41.7728, -87.6650),
    "West Garfield Park": (41.8802, -87.7244),
    "West Lawn": (41.7728, -87.7244),
    "West Pullman": (41.6828, -87.6324),
    "West Ridge": (41.9984, -87.6950),
    "West Town": (41.8952, -87.6883),
    "Woodlawn": (41.7808, -87.5981),
}


# H3 Data Loading

@lru_cache(maxsize=1)
def _load_h3_data() -> pd.DataFrame:
    h3_path = DATA_DIR / "chirde.h3_features.parquet"
    if h3_path.exists():
        return pd.read_parquet(h3_path)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _get_all_h3_indexes() -> frozenset:
    df = _load_h3_data()
    if "h3_index" in df.columns:
        return frozenset(df["h3_index"].tolist())
    return frozenset()


def _get_hexagons_near(lat: float, lon: float, ring_size: int = 3) -> List[str]:
    """Get hexagons near a lat/lon that exist in our dataset."""
    if not HAS_H3:
        return []

    center = _latlng_to_h3(lat, lon, 8)
    neighborhood = _k_ring(center, ring_size)
    known = _get_all_h3_indexes()
    valid = [h for h in neighborhood if h in known]

    if not valid:
        for r in range(ring_size + 1, ring_size + 5):
            neighborhood = _k_ring(center, r)
            valid = [h for h in neighborhood if h in known]
            if valid:
                break

    return valid if valid else [center]


# Public API

def get_all_area_names() -> List[str]:
    """Get sorted list of all Chicago community area names."""
    return sorted(CHICAGO_COMMUNITY_AREAS.keys())


def get_h3_for_area(area_name: str) -> List[str]:
    """Get H3 hexagons covering a community area."""
    if area_name not in CHICAGO_COMMUNITY_AREAS:
        return []

    lat, lon = CHICAGO_COMMUNITY_AREAS[area_name]
    # Community areas are roughly 1-2 km across
    # ring_size=4 covers about 2km radius at resolution 8
    return _get_hexagons_near(lat, lon, ring_size=4)


def get_h3_count_for_area(area_name: str) -> int:
    return len(get_h3_for_area(area_name))


def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """Convert street address to lat/lon."""
    try:
        import requests as req

        if "chicago" not in address.lower():
            address = f"{address}, Chicago, IL"

        resp = req.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1, "countrycodes": "us"},
            headers={"User-Agent": "ChicagoRideDemand/1.0"},
            timeout=5,
        )

        if resp.status_code == 200 and resp.json():
            result = resp.json()[0]
            return float(result["lat"]), float(result["lon"])
    except Exception as e:
        print(f"Geocoding failed: {e}")

    return None


def get_h3_from_lat_lon(lat: float, lon: float, resolution: int = 8) -> str:
    if HAS_H3:
        return _latlng_to_h3(lat, lon, resolution)
    return ""


def get_h3_neighborhood(lat: float, lon: float, ring_size: int = 2,
                         resolution: int = 8) -> List[str]:
    """Get H3 hexagons near a point that exist in our data."""
    return _get_hexagons_near(lat, lon, ring_size)


def get_hexagons_for_input(
    mode: str,
    address: str = None,
    area_name: str = None,
    h3_index: str = None,
    ring_size: int = 2,
) -> Tuple[List[str], str]:
    """Universal resolver: any input → list of H3 hexagons."""

    if mode == "Street Address" and address:
        coords = geocode_address(address)
        if coords:
            lat, lon = coords
            hexagons = get_h3_neighborhood(lat, lon, ring_size=ring_size)
            center = get_h3_from_lat_lon(lat, lon)
            desc = (f"📍 {address}\n"
                    f"Resolved: ({lat:.4f}, {lon:.4f})\n"
                    f"Center hex: {center}\n"
                    f"Coverage: {len(hexagons)} hexagons (ring={ring_size})")
            return hexagons, desc
        return [], f"❌ Could not geocode: {address}"

    elif mode == "Community Area" and area_name:
        hexagons = get_h3_for_area(area_name)
        lat, lon = CHICAGO_COMMUNITY_AREAS.get(area_name, (0, 0))
        desc = (f"🏘️ {area_name}\n"
                f"Center: ({lat:.4f}, {lon:.4f})\n"
                f"Coverage: {len(hexagons)} hexagons")
        return hexagons, desc

    elif mode == "Manual H3" and h3_index:
        if HAS_H3:
            neighbors = list(_k_ring(h3_index, ring_size))
            known = _get_all_h3_indexes()
            valid = [h for h in neighbors if h in known]
            if not valid:
                valid = [h3_index]
            desc = f"🔷 {h3_index} + {len(valid)-1} neighbors"
            return valid, desc
        return [h3_index], f"🔷 {h3_index}"

    return [], "No input provided"