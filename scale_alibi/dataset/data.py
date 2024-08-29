from geojson_pydantic.geometries import Polygon
from pydantic import BaseModel

class Image(BaseModel):
    # this will have... other metadata eventually

    polygon: Polygon

