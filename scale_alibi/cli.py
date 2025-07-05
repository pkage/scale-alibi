from textwrap import dedent
import asyncclick as click
from dotenv import load_dotenv
import mercantile
import pendulum
import numpy as np
from pathlib import Path
from rich.table import Table
from rich import box
import torch.multiprocessing as mp
from torch import cuda
import torch.distributed as dist

from .train import CromaParams, ScaleAlibiParams, TrainParams, cleanup, croma_train, salibi_train

from . import console, pprint, track
from .dataset.search import create_sar_script, create_visual_script, get_sar_images, get_visual_images, create_scl_script
from .dataset.tile import (
    convert_to_png_sar_tiles,
    convert_to_png_scl_tiles,
    convert_to_png_tiles,
    create_downsamples,
    create_zoom_list,
    repair_broken_tiles,
    get_tile_list,
    get_tile_list_all,
    merge_tilesets,
    remove_alpha_tiles,
    tileid_to_zxy
)
from .util import split_to_tuple

def parse_tile(tile_str: str) -> mercantile.Tile:
    z,x,y = tile_str.split('/')
    return mercantile.Tile(z=int(z), x=int(x), y=int(y))

# pre-work
load_dotenv()

@click.group()
def cli():
    pass

@cli.command('debug', help='debug hook')
@click.option('-i', '--input', type=click.Path(readable=True), help='input archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output numpy array', required=True)
def cli_debug(input, output):
    from pmtiles.reader import MmapSource
    from pmtiles.tile import deserialize_header
    from .dataset.pmtile import all_tile_entries

    console.log(f'beginning read of {input}...')

    get_bytes = MmapSource(open(input, 'rb'))
    header = deserialize_header(get_bytes(0,127))

    pprint(header)

    tiles = []
    with console.status('reading...'):
        for tile,_, size in all_tile_entries(get_bytes):
            tiles.append((tile, size))

    tiles = np.array(tiles)

    console.log(f'found {tiles.shape[0]} tiles')

    sizes = tiles[:,1]

    console.log(f'mean: {np.mean(sizes)} bytes ({np.mean(sizes)/1024:2} kib)')
    console.log(f'medn: {np.median(sizes)} bytes ({np.mean(sizes)/1024:2} kib)')

    console.log(f'min: {np.min(sizes)} bytes ({np.min(sizes)/1024:2} kib)')
    console.log(f'max: {np.max(sizes)} bytes ({np.max(sizes)/1024:2} kib)')

    for q in [0, .2, .4, .6, .8, 1]:
        quantile = np.quantile(sizes, q)
        console.log(f'q[bold cyan]{q:3}[/]: {quantile} bytes ({quantile / 1024:02} kib)')



@cli.command('hardware', help='show hardware status')
def cli_hardware():

    def available(flag):
        if flag:
            return '[green]available[/]'
        return '[red]unavailable[/]'

    table = Table('feature', 'status', show_edge=False)

    table.add_row('cuda', available(cuda.is_available()))
    table.add_row('cuda/devices', f'{cuda.device_count()}')
    table.add_row('dist', available(dist.is_available()))
    table.add_row('dist/nccl', available(dist.is_nccl_available()))
    table.add_row('dist/gloo', available(dist.is_gloo_available()))

    console.print(table)

@cli.group('download')
def download():
    pass

@download.command('sar')
@click.option('-t', '--tiles', type=str, help='tile in Z/X/Y. can accept multiple', multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output script', required=True)
@click.option('-l', '--tile-level', type=int, help='generate tiles at this level', default=15)
def download_sar(tiles, output, tile_level):
    all_images = []
    for tile in tiles:
        tile = parse_tile(tile)

        images = get_sar_images(
            tile,
            pendulum.now().subtract(weeks=4)
        )

        all_images += images

    script = create_sar_script(all_images, tile_level)

    with open(output, 'w') as fp:
        fp.write(script)


@download.command('visual')
@click.option('-t', '--tiles', type=str, help='tile in Z/X/Y. can accept multple', multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output script', required=True)
@click.option('-l', '--tile-level', type=int, help='generate tiles at this level', default=15)
def download_visual(tiles, output, tile_level):
    all_images = []
    for tile in tiles:
        tile = parse_tile(tile)

        images = get_visual_images(
            tile,
            pendulum.now().subtract(weeks=4)
        )

        all_images += images

    script = create_visual_script(all_images, tile_level)

    with open(output, 'w') as fp:
        fp.write(script)


@download.command('scl')
@click.option('-t', '--tiles', type=str, help='tile in Z/X/Y. can accept multple', multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output script', required=True)
@click.option('-l', '--tile-level', type=int, help='generate tiles at this level', default=15)
def download_scl(tiles, output, tile_level):
    all_images = []
    for tile in tiles:
        tile = parse_tile(tile)

        images = get_visual_images(
            tile,
            pendulum.now().subtract(weeks=4)
        )

        all_images += images

    script = create_scl_script(all_images, tile_level)

    with open(output, 'w') as fp:
        fp.write(script)

@cli.group('raster')
def raster():
    pass


@raster.command('tile-visual', help='create pmtile archive from a tiff file')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tiff files', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to create tiles at', default=17)
def raster_process_geotiff_visual(input, output, level):
    console.log(input, output, level)
    
    # get_tile_schedule(input, min_zoom=level, max_zoom=level+1, quiet=False)
    convert_to_png_tiles(input, output, min_zoom=level, max_zoom=level+1)

@raster.command('tile-sar', help='create pmtile archive from a tiff file (sar flavored)')
@click.option('-vv', '--vv', type=click.Path(readable=True), help='vv band input tiff', required=True)
@click.option('-vh', '--vh', type=click.Path(readable=True), help='vh band input tiff', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to create tiles at', default=17)
def raster_process_geotiff_sar(vv, vh, output, level):
    console.log(vv,vh, output, level)
    
    convert_to_png_sar_tiles(vv, vh, output, min_zoom=level, max_zoom=level+1)

@raster.command('tile-scl', help='create pmtile archive from a tiff file')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tiff files', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to create tiles at', default=17)
def raster_process_geotiff_scl(input, output, level):
    console.log(input, output, level)
    
    # get_tile_schedule(input, min_zoom=level, max_zoom=level+1, quiet=False)
    convert_to_png_scl_tiles(input, output, min_zoom=level, max_zoom=level+1)

# def create_downsamples(filename: str, outfile: str, source_level: Optional[int], final_level: int, resampling: Optional[Resampling] = Resampling.NEAREST):

@raster.command('downsample', help='create a display tileset by downsampling a tile archive')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-l', '--level', type=int, help='Z level to source from', default=17)
@click.option('-f', '--final-level', type=int, default=1, help='final Z level to create')
def raster_downsample(input, output, level, final_level):
    console.log(input, output, level)
    
    create_downsamples(
        input,
        output,
        level,
        final_level
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
@click.option('--all', is_flag=True, help='when set, do not perform deduplication')
def raster_tile_list(input, output, all):
    console.log(input, output)

    if all:
        arr = get_tile_list_all(input)
    else:
        arr = get_tile_list(input)

    console.log(f'found {arr.shape[0]} tiles')
    np.save(output, arr)


@raster.command('info', help='get all tiles in a raster as a list')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True)
def raster_tile_info(input):
    from .dataset.pmtile import tile_get_all_ids
    # calculate tile list
    arr = tile_get_all_ids(input)

    # parse the tile list
    z_indices = {}
    for tileid in track(arr, description='crunching numbers'):
        z, _, _ = tileid_to_zxy(tileid)

        if not z in z_indices:
            z_indices[z] = 1
        else:
            z_indices[z] += 1

    # format for printing
    z_indices = [z for z in z_indices.items()]
    z_indices.sort(key=lambda p: p[0])

    for z, count in  z_indices:
        console.print(f'level [blue]{z}[/]: [green]{count}[/] tiles.')


@raster.command('repack')
@click.option('-i', '--input', type=click.Path(readable=True), help='input archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output numpy array', required=True)
@click.option('-l', '--level', type=int, help='if set, only copy these levels', multiple=True)
def raster_tile_repack(input, output, level):
    from .dataset.pmtile import tile_repack

    tile_repack(input, output, levels=None if len(level) == 0 else level)


@raster.command('reencode')
@click.option('-i', '--input', type=click.Path(readable=True), help='input archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output numpy array', required=True)
def raster_tile_reencode(input, output):
    from .dataset.pmtile import tile_encode_jpeg

    tile_encode_jpeg(input, output)

@raster.command('filter-empty')
@click.option('-i', '--input', type=click.Path(readable=True), help='input archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output archive', required=True)
def raster_tile_filter_empty(input, output):
    from .dataset.pmtile import tile_filter_empty

    tile_filter_empty(input, output)

@raster.command('merge-fast')
@click.option('-i', '--input', type=click.Path(readable=True), help='input archive', required=True, multiple=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output archive', required=True)
@click.option('-l', '--level', type=int, help='if set, only copy these levels', multiple=True)
def raster_tile_merge_fast(input, output, level):
    from .dataset.pmtile import tile_merge_fast

    tile_merge_fast(input, output, levels=None if len(level) == 0 else level)


@raster.command('tile-sample', help='sample a random tile')
@click.option('-i', '--input', type=click.Path(readable=True), help='input archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output file', required=True)
def tile_sample(input, output):
    from pmtiles.reader import MmapSource
    from pmtiles.tile import deserialize_header
    from .dataset.pmtile import all_tile_entries
    import random

    get_bytes = MmapSource(open(input, 'rb'))

    tiles = []
    with console.status('reading...'):
        for tile, offset, size in all_tile_entries(get_bytes):
            tiles.append((tile, offset, size))


    tile_id, offset, size = random.choice(tiles)
    z, x, y = tileid_to_zxy(tile_id)
    console.print(f'selected [bold cyan]{z}/{x}{y}[/] from {len(tiles)} candidates')

    with open(output, 'wb') as fp:
        fp.write(get_bytes(offset, size))

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
        n_workers=workers
    )


@raster.command('tile-download-split')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile list (npy)', required=True)
@click.option('-a', '--archive', type=click.Path(writable=True), help='archive name', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output download script', required=True)
@click.option('-n', '--n-splits', type=int, help='split size', default=16)
@click.option('-m', '--n-merges', type=int, help='merge size (useful if you have a lot of splits)', default=16)
@click.option('-w', '--workers', type=int, help='worker count', default=4)
def raster_tile_download_split(input, archive, output, n_splits, n_merges, workers):
    console.log(input, archive, output, n_splits, n_merges)
    input = Path(input)
    archive = Path(archive)

    arr = np.load(input)
    subarrs = np.array_split(arr, n_splits)

    output_filenames = []

    for i, subarr in enumerate(subarrs):
        output_filename = f'{input.stem}_chunk_{i}.npy'
        output_filename = input.with_name(output_filename)
        np.save(output_filename, subarr)
        output_filenames.append(output_filename)

    script = dedent(f'''\
        #! /bin/sh

        echo "downloading {len(output_filenames)} chunks..."
    ''') + '\n\n'

    
    chunk_filenames = []

    for i, output_filename in enumerate(output_filenames):
        chunk_filename = f'{archive.stem}_chunk_{i}.pmtile'
        chunk_filenames.append(chunk_filename)
        script += dedent(f'''\
            if ! [ -s {chunk_filename} ]; then
                echo "{chunk_filename} not found, downloading..."
                echo "downloading chunk {i+1} of {len(output_filenames)}"
                salibi raster tile-download -i {output_filename} -o {chunk_filename} -w {workers}
            else
                echo "{chunk_filename} ({i+1} of {len(output_filenames)}) has been downloaded already"
            fi
        ''') + '\n'


    # calculate the merges
    submerges = np.array_split(chunk_filenames, n_merges)
    submerges = [sm.tolist() for sm in submerges]

    script += dedent(f'''\
        echo "merging {len(output_filenames)} chunks into {len(submerges)} sub-merges..."
    ''') + '\n\n'

    merge_filenames = []
    for i, submerge in enumerate(submerges):
        merge_filename = f'{archive.stem}_merge_{i}.pmtile'
        merge_args = ' '.join([f'-i {chunk}' for chunk in submerge])

        script += dedent(f'''\
            if ! [ -s {merge_filename} ]; then
                echo "{merge_filename} not found, merging..."
                echo "merging chunk {i+1} of {len(submerges)}"
                salibi raster merge {merge_args} -o {merge_filename}
            else
                echo "{merge_filename} ({i+1} of {len(submerges)}) has been merged already"
            fi
        ''') + '\n'

        merge_filenames.append(merge_filename)



    merge_args = ' '.join([f'-i {chunk}' for chunk in merge_filenames])
    script += dedent(f'''\
        echo "merging {len(submerges)} chunks into final bundle..."

        if ! [ -s {archive} ]; then
            echo "final archive {archive} not found, merging..."
            salibi raster merge {merge_args} -o {archive}
        else
            echo "{archive} has been merged already"
        fi
    ''') + '\n'


    script += dedent(f'''\
        echo "done!"
    ''')

    with open(output, 'w') as fp:
        fp.write(script)


    console.log(f'wrote {len(subarrs)} chunks with ~{subarrs[0].shape[0]} tiles each,\nthen {len(submerges)} sub-merge operations ({len(submerges[0])} files each) before final merge')


@raster.command('tile-filter', help='filter out tiles with alpha channels from a tileset')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
@click.option('-m', '--max-alpha', type=float, default=0.05, help='maximum alpha values permitted in image')
def raster_tile_filter(input, output, max_alpha):
    console.log(input, output, max_alpha)

    remove_alpha_tiles(
        input,
        output,
        threshold=max_alpha
    )

@raster.command('tile-repair', help='redownload missing tiles in a dataset')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True), help='output tile archive', required=True)
def raster_tile_repair(input, output):
    console.log(input, output)

    repair_broken_tiles(
        input,
        output,
        'https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/tile/{z}/{y}/{x}?blankTile=false',
    )

@raster.command('tile-split', help='split a pmtile archive into smaller split lists')
@click.option('-i', '--input', type=click.Path(readable=True), help='input tile archive', required=True)
@click.option('-o', '--output', type=click.Path(writable=True, dir_okay=True), help='output folder', required=True)
@click.option('-l', '--levels', type=str, help='allowed tile levels', required=True, multiple=True)
@click.option('-s', '--splits', type=str, help='tile split sizes', required=True)
def tile_sample_split(input, output, levels, splits):
    from .dataset.pmtile import tile_sample
    splits = [float(f) for f in split_to_tuple(splits, separator=',')]
    levels = [int(l) for l in levels]

    input = Path(input)
    output = Path(output)

    output_splits = tile_sample(
        input,
        splits=splits,
        levels=set(levels)
    )

    with console.status('writing arrays...'):
        for i, arr in enumerate(output_splits):
            output_filename = f'split_{i}.npy'

            np.save(
                output / output_filename,
                arr
            )

            console.print(f'wrote {arr.shape[0]} tile ids to {output / output_filename}')


# --- CROMA ---
@cli.group('croma', help='commands for reproducing CROMA results')
def croma():
    pass

@croma.command('train', help='train the full embedder/decoder network')
@click.option('--lores', type=click.Path(readable=True), required=True, help='low resolution visual path (sentinel-2)')
@click.option('--radar', type=click.Path(readable=True), required=True, help='SAR path (sentinel-1)')
@click.option('--ckpts', type=click.Path(dir_okay=True, file_okay=False), required=True, help='path to write checkpoints to')
@click.option('--run-name', type=str, required=True, help='run name')
@click.option('--run-group', type=str, default='croma', help='run group (for wandb)')
@click.option('-l', '--learning-rate', type=float, required=True, help='learning rate (Adam)')
@click.option('-e', '--epochs', type=int, required=True, help='epoch count')
@click.option('-b', '--batch-size', type=int, required=True, help='batch size')
@click.option('-d', '--device', type=click.Choice(['cpu', 'cuda', 'mps']), required=True, help='device to run on')
@click.option('-m', '--mask-ratio', type=float, default=0.4, help='mask ratio (ratio of patches to keep)')
@click.option('--no-resume', is_flag=True, help='always start from scratch')
@click.option('--nccl-bind', type=str, default='tcp://localhost:33445', help='distributed synchronization store')
@click.option('--amp', type=bool, is_flag=True, help='enable automatic mixed precision')
@click.option('--half-resolution', type=bool, is_flag=True, help='train at half resolution to conserve vram')
def cli_croma_train( # rename so it doesn't clash
        lores,
        radar,
        ckpts,
        run_name,
        run_group,
        learning_rate,
        epochs,
        batch_size,
        device,
        mask_ratio,
        no_resume,
        nccl_bind,
        amp,
        half_resolution
    ):

    ckpts = Path(ckpts)
    ckpts.mkdir(exist_ok=True, parents=True)

    croma_params = CromaParams(
        lores_dataset_path=Path(lores),
        radar_dataset_path=Path(radar),

        learning_rate=learning_rate,
        batch_size=batch_size,
        mask_ratio=mask_ratio,
        epochs=epochs,
        half_resolution=half_resolution
    )
    train_params = TrainParams(
        checkpoint_dir=ckpts,
        run_name=run_name,
        device=device,
        amp=amp,
        nccl_bind=nccl_bind,
        resume=not no_resume # resume by default
    )


    console.print('parsed parameters:')
    pprint(croma_params)

    world_size = cuda.device_count()
    console.print(f'using world size of {world_size}')


    mp.spawn(
        croma_train,
        (
            world_size,
            croma_params,
            train_params
        ),
        nprocs=world_size,
        join=True
    )

    cleanup()


# --- salibi ---

@cli.command('train', help='train the full embedder/decoder network')
@click.option('--lores', type=click.Path(readable=True), required=True, help='low resolution visual path (sentinel-2)')
@click.option('--radar', type=click.Path(readable=True), required=True, help='SAR path (sentinel-1)')
@click.option('--hires', type=click.Path(readable=True), required=True, help='high resolution visual (naip tiles)')
@click.option('--ckpts', type=click.Path(dir_okay=True, file_okay=False), required=True, help='path to write checkpoints to')
@click.option('--flist', type=click.Path(readable=True), required=False, default=None, help='list of tiles to filter down to (optional)')
@click.option('--run-name', type=str, required=True, help='run name')
@click.option('--run-group', type=str, default='croma', help='run group (for wandb)')
@click.option('-l', '--learning-rate', type=float, required=True, help='learning rate (Adam)')
@click.option('-e', '--epochs', type=int, required=True, help='epoch count')
@click.option('-b', '--batch-size', type=int, required=True, help='batch size')
@click.option('--batch-limit', type=int, help='batch limit per epoch', default=None)
@click.option('-d', '--device', type=click.Choice(['cpu', 'cuda', 'mps']), required=True, help='device to run on')
@click.option('-m', '--mask-ratio', type=float, default=0.4, help='mask ratio (ratio of patches to keep)')
@click.option('--patch-size', type=int, default=16, help='side length of patches to make')
@click.option('--patch-count', type=int, default=256, help='number of patches to target')
@click.option('--no-resume', is_flag=True, help='always start from scratch')
@click.option('--nccl-bind', type=str, default='tcp://localhost:33445', help='distributed synchronization store')
@click.option('--amp', type=bool, is_flag=True, help='enable automatic mixed precision')
@click.option('--half-resolution', type=bool, is_flag=True, help='train at half resolution to conserve vram')
def cli_salibi_train(
        lores,
        radar,
        hires,
        ckpts,
        flist,
        run_name,
        run_group,
        learning_rate,
        epochs,
        batch_size,
        batch_limit,
        device,
        mask_ratio,
        patch_size,
        patch_count,
        no_resume,
        nccl_bind,
        amp,
        half_resolution
    ):

    ckpts = Path(ckpts)
    ckpts.mkdir(exist_ok=True, parents=True)

    salibi_params = ScaleAlibiParams(
        lores_dataset_path=Path(lores),
        radar_dataset_path=Path(radar),
        hires_dataset_path=Path(hires),
        tile_filter_list_path=Path(flist) if flist is not None else None,

        learning_rate=learning_rate,
        batch_size=batch_size,
        batch_limit=batch_limit,
        mask_ratio=mask_ratio,
        epochs=epochs,

        patch_size=patch_size,
        num_patches=patch_count,
        half_resolution=half_resolution
    )
    train_params = TrainParams(
        checkpoint_dir=ckpts,
        run_name=run_name,
        device=device,
        amp=amp,
        nccl_bind=nccl_bind,
        resume=not no_resume
    )


    console.print('parsed parameters:')
    pprint(salibi_params)

    world_size = cuda.device_count()
    console.print(f'using world size of {world_size}')


    mp.spawn(
        salibi_train,
        (
            world_size,
            salibi_params,
            train_params
        ),
        nprocs=world_size,
        join=True
    )

    cleanup()

