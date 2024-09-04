import asyncclick as click
from dotenv import load_dotenv
import mercantile
import pendulum
import numpy as np

from . import console
from .dataset.search import create_sar_script, create_visual_script, get_sar_images, get_visual_images
from .dataset.tile import (
    convert_to_png_sar_tiles,
    convert_to_png_tiles,
    create_downsamples,
    create_zoom_list,
    get_tile_list,
    merge_tilesets,
)

def parse_tile(tile_str: str) -> mercantile.Tile:
    z,x,y = tile_str.split('/')
    return mercantile.Tile(z=int(z), x=int(x), y=int(y))

# pre-work
load_dotenv()

@click.group()
def cli():
    pass

@cli.command('debug', help='debug hook')
def cli_debug():
    ...

@cli.group('download')
def download():
    pass

@download.command('sar')
@click.option('-t', '--tiles', type=str, help='tile in Z/X/Y. can accept multiple', multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output script', required=True)
def download_sar(tiles, output):
    all_images = []
    for tile in tiles:
        tile = parse_tile(tile)

        images = get_sar_images(
            tile,
            pendulum.now().subtract(weeks=4)
        )

        all_images += images

    script = create_sar_script(all_images)

    with open(output, 'w') as fp:
        fp.write(script)

@download.command('visual')
@click.option('-t', '--tiles', type=str, help='tile in Z/X/Y. can accept multple', multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output script', required=True)
def download_visual(tiles, output):
    all_images = []
    for tile in tiles:
        tile = parse_tile(tile)

        images = get_visual_images(
            tile,
            pendulum.now().subtract(weeks=4)
        )

        all_images += images

    script = create_visual_script(all_images)

    with open(output, 'w') as fp:
        fp.write(script)


@cli.group('raster')
def raster():
    pass


@raster.command('tile-visual', help='create pmtile archive from one or more archives')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tiff files', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to create tiles at', default=17)
def raster_process_geotiff_visual(input, output, level):
    console.log(input, output, level)
    
    # get_tile_schedule(input, min_zoom=level, max_zoom=level+1, quiet=False)
    convert_to_png_tiles(input, output, min_zoom=level, max_zoom=level+1)

@raster.command('tile-sar', help='create pmtile archive from one or more archives (sar flavored)')
@click.option('-vv', '--vv', type=click.Path(readable=True), help='vv band input tiff', required=True)
@click.option('-vh', '--vh', type=click.Path(readable=True), help='vh band input tiff', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to create tiles at', default=17)
def raster_process_geotiff_sar(vv, vh, output, level):
    console.log(vv,vh, output, level)
    
    convert_to_png_sar_tiles(vv, vh, output, min_zoom=level, max_zoom=level+1)


# def create_downsamples(filename: str, outfile: str, source_level: Optional[int], final_level: int, resampling: Optional[Resampling] = Resampling.NEAREST):

@raster.command('downsample', help='create pmtile archive from one or more archives')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to source from', default=17)
def raster_downsample(input, output, level):
    console.log(input, output, level)
    
    create_downsamples(
        input,
        output,
        level,
        1
    )

@raster.command('merge', help='merge pmtile archives into one')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True, multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
def raster_merge(input, output):
    console.log(input, output)

    merge_tilesets(
        input,
        output
    )

@raster.command('tile-list', help='get all tiles in a raster as a list')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True, multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile list (npy)', required=True)
def raster_tile_list(input, output):
    console.log(input, output)

    arr = get_tile_list(input)
    console.log(f'found {arr.shape[0]} tiles')
    np.save(output, arr)


@raster.command('tile-zoom', help='get all tiles in a raster as a list')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile list (npy)', required=True)
@click.option('-s', '--source-level', help='source_level', required=True, type=int)
@click.option('-l', '--levels', type=int, help='input tile list (npy)', required=True, multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile list (npy)', required=True)
def raster_tile_zoom(input, source_level, levels, output):
    console.log(input, source_level, levels, output)

    arr = np.load(input)
    console.log(f'found {arr.shape[0]} tiles')

    zoom = create_zoom_list(
        arr,
        source_level,
        levels
    )
    console.log(f'created {zoom.shape[0]} tiles')

    np.save(output, zoom)

    # arr = get_tile_list(input)
    # console.log(f'found {arr.shape[0]} tiles')
    # np.save(output, arr)

@raster.command('tile-download')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile list (npy)', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-w', '--workers', type=int, help='number of workers', default=2)
def raster_tile_download(input, output, workers):
    console.log(input, output)

    arr = np.load(input)
    console.log(f'found {arr.shape[0]} tiles')

    from .dataset.download import download_tile_archive

    download_tile_archive(
        arr,
        'https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/tile/{z}/{y}/{x}?blankTile=false',
        output,
        workers=workers
    )
