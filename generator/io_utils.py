"""
I/O utilities: geometry serialization to JSON, JSONL write, sample meta.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry


def to_json_geom(geom: BaseGeometry) -> Dict[str, Any]:
    """Serialize Shapely geometry to a JSON-serializable dict."""
    if isinstance(geom, Point):
        return {"type": "point", "coords": [geom.x, geom.y]}
    if isinstance(geom, LineString):
        return {
            "type": "polyline",
            "coords": [[float(x), float(y)] for x, y in geom.coords],
        }
    if isinstance(geom, Polygon):
        exterior = [[float(x), float(y)] for x, y in geom.exterior.coords]
        holes = [
            [[float(x), float(y)] for x, y in ring.coords]
            for ring in geom.interiors
        ]
        return {"type": "polygon", "coords": exterior, "holes": holes}
    if isinstance(geom, MultiPolygon):
        return {
            "type": "multipolygon",
            "parts": [to_json_geom(g) for g in geom.geoms],
        }
    raise ValueError(f"Unsupported geometry type: {type(geom)}")


def geom_meta(geom: BaseGeometry) -> Dict[str, Any]:
    """Basic meta for one geometry (vertices, has_hole, etc.)."""
    meta: Dict[str, Any] = {}
    if isinstance(geom, Point):
        meta["vertices"] = 1
        meta["has_hole"] = False
    elif isinstance(geom, LineString):
        meta["vertices"] = len(geom.coords)
        meta["has_hole"] = False
    elif isinstance(geom, Polygon):
        meta["vertices"] = len(geom.exterior.coords) - 1  # closed ring
        meta["has_hole"] = len(geom.interiors) > 0
    elif isinstance(geom, MultiPolygon):
        meta["vertices"] = sum(
            len(g.exterior.coords) - 1 + sum(len(r.coords) for r in g.interiors)
            for g in geom.geoms
        )
        meta["has_hole"] = any(len(g.interiors) > 0 for g in geom.geoms)
    else:
        meta["vertices"] = 0
        meta["has_hole"] = False
    return meta


def write_jsonl(path: str, records: List[Dict], encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
