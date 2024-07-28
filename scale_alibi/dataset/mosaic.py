from typing import List, Tuple, Optional
from shapely import to_geojson, union_all
from shapely.geometry import shape
from .models import Image
from geojson_pydantic import FeatureCollection, MultiPolygon, Polygon
from .. import console


def get_image_union(images: List[Image]) -> MultiPolygon | Polygon:
    # convert all to shapely, union, then convert back into polygons
    group = [ shape(img.polygon) for img in images ]
    group = union_all(group)
    
    if group.geom_type == 'Polygon':
        group = Polygon.parse_raw(to_geojson(group))
    elif group.geom_type == 'MultiPolygon':
        group = MultiPolygon.parse_raw(to_geojson(group))

    return group


def get_group_coverage(target: Polygon, images: List[Image]) -> List[Tuple[float, Image]]:
    group = get_image_union(images)

    coverage = map(
        lambda img: (get_overlap(target, group), img),
        images
    )

    return sorted(coverage, key=lambda p: p[0], reverse=True)


def get_next_image(aoi: Polygon, frozen: List[Image], candidates: List[Image], _depth=0, max_depth=None) -> List[Image]:
    if len(candidates) == 0 or (max_depth is not None and _depth == max_depth):
        return frozen

    console.print(f'[yellow]candidates: {len(candidates)}')

    # add each candidate image to the frozen pool, and check the new overlap with the AOI coverage
    overlaps = []
    for candidate in candidates:
        overlap = get_overlap(
            aoi,
            get_image_union( frozen + [candidate] )
        )
        overlaps.append( (overlap, candidate) )

    # sort by the overlaps
    overlaps.sort(key=lambda p: p[0], reverse=True)

    # print([p[0] for p in overlaps])

    # get our top candidate to add it to the frozen
    overlap, top_candidate = overlaps[0]
    frozen = [*frozen, top_candidate]

    # print(f'lvl {_depth} top candidate: {overlap}')
    console.print(f'[black]image set {_depth} (max {max_depth}), overlap: {overlap:.02%}')

    # early termination: if we've found a spanning polygon set then return it
    if overlap == 1 :
        return frozen

    # remove overlap info 
    new_candidates = [img for _, img in overlaps[1:]]
    
    # recurse!
    return get_next_image(aoi, frozen, new_candidates, _depth=_depth+1, max_depth=max_depth)


def min_spanning_images(aoi: Polygon, images: List[Image], max_solution_size=None) -> List[Image]:
    if len(images) == 0:
        return []

    sorted_images = sorted(images, key=lambda k: k.aoi_coverage, reverse=True)

    lead_image = sorted_images[0]

    if max_solution_size is None:
        max_solution_size = len(images)
    else:
        # accounts for the frozen image already being in the set
        max_solution_size -= 1

    return get_next_image(aoi, [lead_image], sorted_images[1:], max_depth=max_solution_size)


def get_overlap(poly1: Polygon, poly2: Polygon) -> float:
    '''
    Calculate the overlap between two polygons, from 0 (not overlapping) to 1 (completely overlapping)

    :param poly1: First polygon
    :type poly1: Polygon
    :param poly2: Second polygon
    :type poly2: Polygon
    :return: % overlap between 0 and 1
    :rtype: float
    '''
    # convert the polygons
    spoly1 = shape(poly1.dict())
    spoly2 = shape(poly2.dict())

    # if we don't overlap at all, panic
    if not spoly1.intersects(spoly2):
        return 0.0

    if spoly1.within(spoly2):  #or spoly2.within(spoly1):
        # print('--- polygon overlap ---')
        return 1.0

    # get the intersection %
    intersect = spoly1.intersection(spoly2).area/spoly1.area

    if intersect > 1:
        intersect = 1

    return intersect


def format_all_polys(aoi: Optional[Polygon], images: List[Image]) -> FeatureCollection:
    def wrap_poly(poly, image_id=None):
        return {
            'type': 'Feature',
            'properties': {
                'image_id': image_id
            },
            'geometry': poly.dict(),
            'id': image_id
        }

    obj = {
        'type': 'FeatureCollection',
        'features': [wrap_poly(i.polygon, image_id=i.image_id) for i in images]
    }


    if aoi is not None:
        aoi_poly = wrap_poly(aoi, image_id='aoi')
        aoi_poly['properties'] = {
            "stroke": "#ff0000",
            "stroke-width": 2,
            "stroke-opacity": 1,
            "fill": "#555555",
            "fill-opacity": 0.2
        }
        obj['features'].append(aoi_poly)

    return FeatureCollection.parse_obj(obj)
    # with open('group.json', 'w') as fp:
    #     json.dump(obj, fp, indent=4)

