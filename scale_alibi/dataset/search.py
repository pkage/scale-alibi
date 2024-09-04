from typing import List
from pystac.item import Item

from .data import Image
from .mosaic import min_spanning_images
from .. import pprint, track
from pystac_client import Client
import mercantile
from pendulum import DateTime
from geojson_pydantic.geometries import Polygon

from textwrap import dedent

STAC_URL = 'https://earth-search.aws.element84.com/v1'

def image_search(collection: str, tile: mercantile.Tile, time_start: DateTime, weeks: int = 2, searches: int = 1) -> List[Item]:
    bbox = mercantile.bounds(tile)

    catalog = Client.open(STAC_URL)

    # annoyingly this aws opendata endpoint doesn't support filtering
    # so we gotta do it ourselves. we're also breaking up the searching
    # into multiple calls so as not to crash the endpoint by returning too many results
    results = []
    for _ in track(range(searches), description='searching...'):
        time_end = time_start.add(weeks=weeks)

        items = catalog.search(
            collections=[collection],
            max_items=20,
            # bbox=[-73.21, 43.99, -73.12, 44.05],
            bbox=[bbox.west, bbox.south, bbox.east, bbox.north],
            datetime=[
                time_start,
                time_end
            ]
        ).items()

        results += list(items)

        time_start = time_start.subtract(weeks=weeks)

    return results

def get_best_coverage(tile: mercantile.Tile, items: List[Item]) -> List[Item]:
    # prep image list
    image_list = []
    for item in items:
        image_list.append(
            Image(
                polygon=Polygon.parse_obj(item.geometry),
                item=item
            )
        )

    target_polygon = Polygon.parse_obj(
        mercantile.feature(tile)['geometry']
    )

    filtered_images = min_spanning_images(
        target_polygon,
        image_list
    )

    pprint(filtered_images)

    return [i.item for i in filtered_images]


def get_sar_images(tile: mercantile.Tile, time_start: DateTime, weeks: int = 2):
    results = image_search(
        'sentinel-1-grd',
        tile,
        time_start,
        weeks=weeks,
        searches=5
    )

    print(f'all results: {len(results)}')

    filtered_results: List[Item] = []
    for item in results:
        if item.properties['sar:frequency_band'] != 'C':
            continue
        if item.properties['sar:instrument_mode'] != 'IW':
            continue

        filtered_results.append(item)
        
    print(f'filtered down to {len(filtered_results)}/{len(results)} items')

    print(filtered_results[0].geometry)

    coverage_filter = get_best_coverage(tile, filtered_results)
    print('coverage filter', coverage_filter)

    return coverage_filter

def get_visual_images(tile: mercantile.Tile, time_start: DateTime, weeks: int = 2):
    results = image_search(
        'sentinel-2-l2a',
        tile,
        time_start,
        weeks=weeks,
        searches=8
    )

    print(f'all results: {len(results)}')

    filtered_results: List[Item] = []
    for item in results:
        if item.properties['eo:cloud_cover'] > 10: # percent
            continue

        filtered_results.append(item)
        
    print(f'filtered down to {len(filtered_results)}/{len(results)} items')

    print(filtered_results[0].geometry)

    coverage_filter = get_best_coverage(tile, filtered_results)
    print('coverage filter', coverage_filter)

    return coverage_filter


def create_sar_script(items: List[Item]):
    def create_download_process_script(item: Item):
        item_name = item.id
        vv_filename = f'{item_name}-vv.tif'
        vh_filename = f'{item_name}-vh.tif'
        tile_filename = f'{item_name}.pmtile'

        vv_url = item.assets['vv'].href
        vh_url = item.assets['vh'].href
        download_item = dedent(f'''\
            if ! [ -f ./rasters/{vv_filename} ]; then
                echo "{vv_filename} not found, downloading..."
                aws s3 cp {vv_url} ./rasters/{vv_filename} --request-payer requester
            else
                echo "{vv_filename} has been downloaded already"
            fi
            if ! [ -f ./rasters/{vh_filename} ]; then
                echo "{vh_filename} not found, downloading..."
                aws s3 cp {vh_url} ./rasters/{vh_filename} --request-payer requester
            else
                echo "{vh_filename} has been downloaded already"
            fi
        ''')

        process_item = dedent(f'''\
            if ! [ -f ./tiles/{tile_filename} ]; then
                salibi raster tile-sar -vv ./rasters/{vv_filename} -vh ./rasters/{vh_filename} -o ./tiles/{tile_filename} -l 15
            else
                echo "{tile_filename} has been generated already"
            fi
        ''')

        return download_item, process_item, tile_filename
    
    header = dedent('''\
        #! /bin/sh

        mkdir -p rasters
        mkdir -p tiles
    ''')

    download_items = []
    process_items = []
    tile_files = []

    for item in items:
        download, process, tile_filename = create_download_process_script(item)

        download_items.append(download)
        process_items.append(process)
        tile_files.append(tile_filename)

    script = header + '\n\necho "---- DOWNLOADING ---"\n\n'

    for i, chunk in enumerate(download_items):
        script += f'echo "downloading {i+1} of {len(download_items)}"\n'
        script += chunk

    script += '\n\necho "---- PROCESSING ---"\n\n'

    for i, chunk in enumerate(process_items):
        script += f'echo "processing {i+1} of {len(download_items)}"\n'
        script += chunk

    script += '\n\necho "---- FINALIZING ---"\n\n'

    # merges
    script += 'echo "merging tiles..."\n'
    
    merged_tile_name = 'sar_tiles.pmtile'
    tile_inputs = ' '.join([f'-i ./tiles/{fn}' for fn in tile_files])

    script +=  dedent(f'''\
        if ! [ -f ./{merged_tile_name} ]; then
            salibi raster merge {tile_inputs} -o ./{merged_tile_name}
        else
            echo "merged tileset {merged_tile_name} has been generated already"
        fi
    ''')

    script += 'echo "creating display tileset..."\n'
    
    downsample_tile_name = 'sar_tiles_display.pmtile'

    script +=  dedent(f'''\
        if ! [ -f ./{downsample_tile_name} ]; then
            salibi raster downsample -i ./{merged_tile_name} -o ./{downsample_tile_name} -l 15
        else
            echo "downsampled tileset {downsample_tile_name} has been generated already"
        fi
    ''')
    

    return script


def create_visual_script(items: List[Item]):
    def create_download_process_script(item: Item):
        item_name = item.id
        visual_filename = f'{item_name}-visual.tif'
        tile_filename = f'{item_name}.pmtile'

        visual_url = item.assets['visual'].href
        download_item = dedent(f'''\
            if ! [ -f ./rasters/{visual_filename} ]; then
                echo "{visual_filename} not found, downloading..."
                curl -L -J {visual_url} -o ./rasters/{visual_filename}
            else
                echo "{visual_url} has been downloaded already"
            fi
        ''')

        process_item = dedent(f'''\
            if ! [ -f ./tiles/{tile_filename} ]; then
                salibi raster tile-visual -i ./rasters/{visual_filename}  -o ./tiles/{tile_filename} -l 15
            else
                echo "{tile_filename} has been generated already"
            fi
        ''')

        return download_item, process_item, tile_filename
    
    header = dedent('''\
        #! /bin/sh

        mkdir -p rasters
        mkdir -p tiles
    ''')

    download_items = []
    process_items = []
    tile_files = []

    for item in items:
        download, process, tile_filename = create_download_process_script(item)

        download_items.append(download)
        process_items.append(process)
        tile_files.append(tile_filename)

    script = header + '\n\necho "---- DOWNLOADING ---"\n\n'

    for i, chunk in enumerate(download_items):
        script += f'echo "downloading {i+1} of {len(download_items)}"\n'
        script += chunk

    script += '\n\necho "---- PROCESSING ---"\n\n'

    for i, chunk in enumerate(process_items):
        script += f'echo "processing {i+1} of {len(download_items)}"\n'
        script += chunk

    script += '\n\necho "---- FINALIZING ---"\n\n'

    # merges
    script += 'echo "merging tiles..."\n'
    
    merged_tile_name = 'visual_tiles.pmtile'
    tile_inputs = ' '.join([f'-i ./tiles/{fn}' for fn in tile_files])

    script +=  dedent(f'''\
        if ! [ -f ./{merged_tile_name} ]; then
            salibi raster merge {tile_inputs} -o ./{merged_tile_name}
        else
            echo "merged tileset {merged_tile_name} has been generated already"
        fi
    ''')

    script += 'echo "creating display tileset..."\n'
    
    downsample_tile_name = 'visual_tiles_display.pmtile'

    script +=  dedent(f'''\
        if ! [ -f ./{downsample_tile_name} ]; then
            salibi raster downsample -i ./{merged_tile_name} -o ./{downsample_tile_name} -l 15
        else
            echo "downsampled tileset {downsample_tile_name} has been generated already"
        fi
    ''')
    

    return script
