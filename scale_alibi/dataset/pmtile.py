# quick and dirty: we want to quickly stat the files here to make sure everything makes sense

import io
import os
from statistics import mean

import numpy as np
from PIL import Image
from pmtiles.reader import MmapSource
from pmtiles.reader import Reader as PMReader
from pmtiles.tile import (
    Compression,
    TileType,
    deserialize_directory,
    deserialize_header,
    tileid_to_zxy,
    zxy_to_tileid,
)
from pmtiles.writer import Writer as PMWriter

from .. import console, track



def traverse_entries(get_bytes, header, dir_offset, dir_length):
    entries = deserialize_directory(get_bytes(dir_offset, dir_length))
    for entry in entries:
        if entry.run_length > 0:
            for i in range(entry.run_length):
                yield entry.tile_id + i, header["tile_data_offset"] + entry.offset, entry.length
        else:
            for t in traverse_entries(
                get_bytes,
                header,
                header["leaf_directory_offset"] + entry.offset,
                entry.length,
            ):
                yield t


def all_tile_entries(get_bytes):
    header = deserialize_header(get_bytes(0, 127))
    return traverse_entries(get_bytes, header, header["root_offset"], header["root_length"])

def tile_get_all_ids(filename):
    get_bytes = MmapSource(open(filename, 'rb'))

    with console.status('reading tile directory...'):
        tile_ids = [t[0] for t in all_tile_entries(get_bytes)]

    return tile_ids



def tile_repack(infile, outfile, levels=None):
    # we're just gonna recreate the file here, so copy those headers
    get_bytes = MmapSource(open(infile, 'rb'))

    # copy manually
    header = deserialize_header(get_bytes(0, 127))
    
    # extract the basic information from the header for the writer
    pminfo = {
        'tile_type': header['tile_type'],
        'tile_compression': header['tile_compression'],

        'min_zoom': header['min_zoom'],
        'max_zoom': header['max_zoom'],

        'min_lon_e7': header['min_lon_e7'],
        'min_lat_e7': header['min_lat_e7'],
        'max_lon_e7': header['max_lon_e7'],
        'max_lat_e7': header['max_lat_e7'],

        'center_zoom': header['center_zoom'],
        'center_lat_e7': header['center_lat_e7'],
        'center_lon_e7': header['center_lon_e7']
    }

    tiles = []
    with console.status('reading inputs...'):
        for tile, offset, size in all_tile_entries(get_bytes):
            tiles.append((tile, offset, size))

    if levels is not None:
        levels = set(levels)
        tiles_filtered = []
        for tile in track(tiles, description='applying filter...'):
            z, _, _ = tileid_to_zxy(tile[0])
            if z in levels:
                tiles_filtered.append(tile)
    
        console.print(f'will only copy {len(tiles_filtered)}/{len(tiles)} tiles ({len(tiles_filtered)/len(tiles):%})')
        tiles = tiles_filtered

    if len(tiles) == 0:
        console.print('[yellow]no tiles specified, aborting...')
        return

    with open(outfile, 'wb') as out_f:
        writer = PMWriter(out_f)

        for tileid, offset, length in track(tiles, description='Creating tiles...', console=console):
            tile_bytes = get_bytes(offset, length)

            writer.write_tile(tileid, tile_bytes)

        with console.status('finalizing...'):
            writer.finalize(
                pminfo,
                {
                    'attribution': 'raster gen'
                }
            )

    osize = os.path.getsize(outfile)
    console.print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')


def encode_single_jpeg(png_bytes):
    with Image.open(io.BytesIO(png_bytes)) as img:
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')

        # Save to JPEG in-memory
        jpeg_buffer = io.BytesIO()
        img.save(jpeg_buffer, format='JPEG', quality=90)  # adjust quality as needed
        return jpeg_buffer.getvalue()


def tile_encode_jpeg(infile, outfile):
    # we're just gonna recreate the file here, so copy those headers
    get_bytes = MmapSource(open(infile, 'rb'))

    # copy manually
    header = deserialize_header(get_bytes(0, 127))
    
    # extract the basic information from the header for the writer
    pminfo = {
        'tile_type': header['tile_type'],
        'tile_compression': header['tile_compression'],

        'min_zoom': header['min_zoom'],
        'max_zoom': header['max_zoom'],

        'min_lon_e7': header['min_lon_e7'],
        'min_lat_e7': header['min_lat_e7'],
        'max_lon_e7': header['max_lon_e7'],
        'max_lat_e7': header['max_lat_e7'],

        'center_zoom': header['center_zoom'],
        'center_lat_e7': header['center_lat_e7'],
        'center_lon_e7': header['center_lon_e7']
    }

    tiles = []
    with console.status('reading inputs...'):
        for tile, offset, size in all_tile_entries(get_bytes):
            tiles.append((tile, offset, size))

    if len(tiles) == 0:
        console.print('[yellow]no tiles specified, aborting...')
        return

    with open(outfile, 'wb') as out_f:
        writer = PMWriter(out_f)

        for tileid, offset, length in track(tiles, description='Creating jpeg tiles...', console=console):
            tile_bytes = get_bytes(offset, length)

            tile_bytes = encode_single_jpeg(tile_bytes)

            writer.write_tile(tileid, tile_bytes)

        with console.status('finalizing...'):
            writer.finalize(
                pminfo,
                {
                    'attribution': 'raster gen'
                }
            )

    osize = os.path.getsize(outfile)
    console.print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')


def tile_merge_fast(infiles, outfile, levels=None):
    # we're just gonna recreate the file here, so copy those headers
    mmaps = []
    all_headers = []
    for infile in infiles:
        get_bytes = MmapSource(open(infile, 'rb'))
        header = deserialize_header(get_bytes(0, 127))
        all_headers.append(header)
        mmaps.append(get_bytes)

    # helper function to get the metadata value
    def get_header_key(headers, key):
        out = []
        for header in headers:
            out.append(header[key])
        return out

    # copy manually
    
    # extract the basic information from the header for the writer
    pminfo = {
        # todo: check if this holds across all tile types
        # we should check and transcode if we're not ready
        'tile_type': all_headers[0]['tile_type'],
        'tile_compression': Compression.NONE,

        # get from bounds
        'min_zoom': min(get_header_key(all_headers, 'min_zoom')),
        'max_zoom': max(get_header_key(all_headers, 'max_zoom')),

        'min_lon_e7': min(get_header_key(all_headers, 'min_lon_e7')),
        'min_lat_e7': min(get_header_key(all_headers, 'min_lat_e7')),
        'max_lon_e7': max(get_header_key(all_headers, 'max_lon_e7')),
        'max_lat_e7': max(get_header_key(all_headers, 'max_lat_e7')),

        # calculating on the fly
        'center_zoom': int(mean([
            min(get_header_key(all_headers, 'min_zoom')),
            max(get_header_key(all_headers, 'max_zoom'))
        ]))
    }
    # this stuff here needs to be in there but is calculated after the fact
    # to avoid some nasty duplication of work
    pminfo['center_lat_e7'] = int(mean([
        pminfo['min_lat_e7'],
        pminfo['max_lat_e7']
    ]))
    pminfo['center_lon_e7'] = int(mean([
        pminfo['min_lon_e7'],
        pminfo['max_lon_e7']
    ]))

    tiles = []
    with console.status('reading inputs...'):
        for i in range(len(mmaps)):
            for tile, offset, size in all_tile_entries(mmaps[i]):
                tiles.append((tile, offset, size, i))

    if levels is not None:
        levels = set(levels)
        tiles_filtered = []
        for tile in track(tiles, description='applying filter...'):
            z, _, _ = tileid_to_zxy(tile[0])
            if z in levels:
                tiles_filtered.append(tile)
    
        console.print(f'will only copy {len(tiles_filtered)}/{len(tiles)} tiles ({len(tiles_filtered)/len(tiles):%})')
        tiles = tiles_filtered


    if len(tiles) == 0:
        console.print('[yellow]no tiles specified, aborting...')
        return

    with console.status('presorting tiles...'):
        tiles.sort(key=lambda t: t[0])

    console.print(f'merging {len(tiles)} tiles from {len(mmaps)} sources')

    with open(outfile, 'wb') as out_f:
        writer = PMWriter(out_f)

        for tileid, offset, length, mmap_id in track(tiles, description='Creating tiles...', console=console):
            tile_bytes = mmaps[mmap_id](offset, length)

            writer.write_tile(tileid, tile_bytes)

        with console.status('finalizing...'):
            writer.finalize(
                pminfo,
                {
                    'attribution': 'raster merge'
                }
            )

    osize = os.path.getsize(outfile)
    console.print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')


def sniff_for_empty_tile(tile_bytes):
    with Image.open(io.BytesIO(tile_bytes)) as img:
        img = np.array(img)

        return (img < 10).all()

def tile_filter_empty(infile, outfile):
    # we're just gonna recreate the file here, so copy those headers
    get_bytes = MmapSource(open(infile, 'rb'))

    # copy manually
    header = deserialize_header(get_bytes(0, 127))
    
    # extract the basic information from the header for the writer
    pminfo = {
        'tile_type': header['tile_type'],
        'tile_compression': header['tile_compression'],

        'min_zoom': header['min_zoom'],
        'max_zoom': header['max_zoom'],

        'min_lon_e7': header['min_lon_e7'],
        'min_lat_e7': header['min_lat_e7'],
        'max_lon_e7': header['max_lon_e7'],
        'max_lat_e7': header['max_lat_e7'],

        'center_zoom': header['center_zoom'],
        'center_lat_e7': header['center_lat_e7'],
        'center_lon_e7': header['center_lon_e7']
    }

    tiles = []
    with console.status('reading inputs...'):
        for tile, offset, size in all_tile_entries(get_bytes):
            tiles.append((tile, offset, size))

    reject_count = 0
    with open(outfile, 'wb') as out_f:
        writer = PMWriter(out_f)

        for tileid, offset, length in track(tiles, description='Creating tiles...', console=console):
            tile_bytes = get_bytes(offset, length)

            if sniff_for_empty_tile(tile_bytes):
                reject_count += 1
            else:
                writer.write_tile(tileid, tile_bytes)

        with console.status('finalizing...'):
            writer.finalize(
                pminfo,
                {
                    'attribution': 'raster gen'
                }
            )

    console.print(f'removed {reject_count}/{len(tiles)} tiles ({reject_count/len(tiles):%})')

    osize = os.path.getsize(outfile)
    console.print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')
