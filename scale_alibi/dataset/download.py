import mercantile
from pmtiles.tile import Compression, TileType, zxy_to_tileid, tileid_to_zxy
from pmtiles.writer import Writer as PMWriter
from io import BytesIO
from PIL import Image
import httpx
from typing import List, Optional, Tuple
from statistics import mean
import os
import asyncio

import concurrent.futures
import multiprocessing

from .. import console, track
from rich import progress as progress
import numpy as np

CONCURRENT_DOWNLOADS_PER_PROCESS = 2

async def download_tile(url: str) -> bytes:
    retry_count = 15

    while True:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url)

            return r.content
        except:
            retry_count -= 1
            if retry_count == 0:
                raise


# async def tile_block_helper(urls: List[str]) -> List[bytes]:
#     tiles = await asyncio.gather(*([ download_tile(url) for url in urls ]))
#     return list(tiles)

def tile_block_helper(tiles: List[mercantile.Tile], url_template: str, progress=None, task_id=None) -> List[Tuple[int, bytes]]:

    # try:
    #     loop = asyncio.get_running_loop()
    # except:
    #     print('no running loop')
    # else:
    #     raise RuntimeError('can\'t have multiple asyncio loops')

    tile_blocks = []
    
    # arrange into blocks
    blocksize = CONCURRENT_DOWNLOADS_PER_PROCESS

    for i in range(1+(len(tiles) // blocksize)):
        # first get the block of tiles we'll be working with
        tile_block = tiles[i*blocksize:(i+1)*blocksize]
        tile_blocks.append(tile_block)

    # download the blocks
    output = []
    loop = asyncio.get_event_loop()
    for i, tile_block in enumerate(tile_blocks):
    
        tile_ids = [zxy_to_tileid(t.z, t.x, t.y) for t in tile_block ]
        urls = [ url_template.format(z=t.z, x=t.x, y=t.y) for t in tile_block ]

        tiles = loop.run_until_complete(
            asyncio.gather(*([ download_tile(url) for url in urls ]))
        )

        output += list(zip(tile_ids, tiles))

        if progress is not None and task_id is not None:
            progress[task_id] = {'progress': i, 'total': len(tile_blocks)}

    if progress is not None and task_id is not None:
        progress[task_id] = {'progress': len(tile_blocks), 'total': len(tile_blocks)}

    return output


def transcode_tile(tile_bytes: bytes, output_format: str) -> bytes:
    # load the image
    # this should work because of the magic numbers in the image file
    image = Image.open(BytesIO(tile_bytes))

    # save the image to a bytes io
    iio = BytesIO()
    image.save(iio, format=output_format)

    # rewind and return the bytes
    iio.seek(0)
    return iio.read()


def download_tile_archive(
        tile_list: np.ndarray,
        url_template: str,
        output: str,
        tiletype: Optional[TileType] = None,
        transcode: bool = False,
        n_workers: Optional[int] = None
    ) -> None:
    '''
    Create a PMTile archive with the in a specific area from disk.

    :param parent_tile: The parent tile to pull from
    :type parent_tile: mercantile.Tile
    :param target_zoom: Target label map zoom.
    :type target_zoom: int

    :param output: Output pmtiles file path
    :type output: str
    '''
    # calculate the list of tiles and pre-sort it
    tile_list = tile_list.flatten()
    tile_list.sort()

    all_tiles = []
    min_zoom = 20
    max_zoom = 0
    for tile_id in tile_list:
        z,x,y = tileid_to_zxy(tile_id)

        if z < min_zoom:
            min_zoom = z
        if z > max_zoom:
            max_zoom = z

        all_tiles.append(
            mercantile.Tile(z=z, x=x, y=y)
        )


    # calculate tile metadata (cheating a bit, metadata will be wrong)
    # z,x,y = tileid_to_zxy(tile_list[0])

    bbox = mercantile.bounds(all_tiles[0])

    to_e7 = lambda n: int(n * 10_000_000)
    pminfo = {
        'tile_type': tiletype if tiletype is not None else TileType.PNG,
        'tile_compression': Compression.NONE,

        'min_zoom': min_zoom,
        'max_zoom': max_zoom, # this might matter

        # these need to be switched because i'm pretty sure pmtiles has it backwards...
        'min_lon_e7': to_e7(bbox.west),
        'min_lat_e7': to_e7(bbox.south),
        'max_lon_e7': to_e7(bbox.east),
        'max_lat_e7': to_e7(bbox.north),

        'center_zoom': int((min_zoom + max_zoom)/2),
        'center_lat_e7': to_e7(mean([bbox.east, bbox.west])),
        'center_lon_e7': to_e7(mean([bbox.north, bbox.south]))
    }

    # download the tiles in blocks
    tile_blocks = []
    blocksize = len(all_tiles) // (os.cpu_count() if os.cpu_count() is not None else 8)

    for i in range(1+(len(all_tiles) // blocksize)):
        # first get the block of tiles we'll be working with
        tile_block = all_tiles[i*blocksize:(i+1)*blocksize]
        tile_blocks.append(tile_block)


    processed_tiles = []

    # code adapted from https://www.deanmontgomery.com/2022/03/24/rich-progress-and-multiprocessing/
    with progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        progress.TimeRemainingColumn(),
        progress.TimeElapsedColumn(),
        refresh_per_second=1,  # bit slower updates
        console=console
    ) as tile_progress:
        futures = []  # keep track of the jobs
        with multiprocessing.Manager() as manager:
            # this is the key - we share some state between our 
            # main process and our worker functions
            _progress = manager.dict()
            overall_progress_task = tile_progress.add_task("[green]All jobs progress:")

            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                for n, tile_block in enumerate(tile_blocks):
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = tile_progress.add_task(f"task {n}/{len(tile_blocks)}", visible=False)
                    futures.append(executor.submit(tile_block_helper, tile_block, url_template, progress=_progress, task_id=task_id))

                # monitor the progress:
                while (n_finished := sum([future.done() for future in futures])) < len(
                    futures
                ):
                    tile_progress.update(
                        overall_progress_task, completed=n_finished, total=len(futures)
                    )
                    for task_id, update_data in _progress.items():
                        latest = update_data["progress"]
                        total = update_data["total"]
                        # update the progress bar for this task:
                        tile_progress.update(
                            task_id,
                            completed=latest,
                            total=total,
                            visible=latest != total
                        )

                # raise any errors and collect the results
                processed_tiles = []
                for future in futures:
                    processed_tiles += future.result()

    

    # make sure these are sorted
    processed_tiles.sort(key=lambda p: p[0])

    # transcode the tiles one at a time for now
    if transcode:
        for i in track(range(len(processed_tiles)), description='Transcoding tiles...'):
            tile_id, tile_bytes = processed_tiles[i]
            processed_tiles[i] = (tile_id, transcode_tile(tile_bytes, 'png'))


    with open(output, 'wb') as out_f:
        writer = PMWriter(out_f)

        for tileid, tile_bytes in track(processed_tiles, description='Writing tiles...'):
            writer.write_tile(tileid, tile_bytes)

        with console.status('Finalizing...'):
            writer.finalize(
                pminfo,
                {
                    'attribution': 'dataset downloader'
                }
            )

    osize = os.path.getsize(output)
    print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')
