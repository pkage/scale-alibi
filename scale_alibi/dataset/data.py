from geojson_pydantic.geometries import Polygon
from pydantic import BaseModel
from pystac import Item
from typing import Any

class Image(BaseModel):
    polygon: Polygon
    item: Any
