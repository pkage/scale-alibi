import asyncclick as click

from scale_alibi.dataset.tile import convert_to_tiles, get_tile_schedule
from . import console
from dotenv import load_dotenv

# pre-work
load_dotenv()

@click.group()
def cli():
    pass

@cli.command('debug', help='debug hook')
def cli_debug():
    console.log('beginning hooks')

@cli.group('dataset')
def dataset():
    ...


@dataset.command('geotiff')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tiff files', required=True, multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to create tiles at', default=17)
def dataset_process_geotiff(input, output, level):
    console.log(input, output, level)
    
    # get_tile_schedule(input, min_zoom=level, max_zoom=level+1, quiet=False)
    # convert_to_tiles(input, output, min_zoom=level, max_zoom=level+1)
