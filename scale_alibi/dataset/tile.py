from collections import namedtuple
from io import BytesIO
import os
from statistics import mean
from typing import Callable, Dict, List, Tuple, Optional, Any

from PIL import Image
from PIL.Image import Resampling
from geojson_pydantic.geometries import Polygon
from geojson_pydantic import Feature
from mercantile import LngLatBbox, Tile, bounding_tile, bounds, children, parent
from pmtiles.tile import Compression, TileType, zxy_to_tileid, tileid_to_zxy
from pmtiles.writer import Writer as PMWriter
from pmtiles.reader import Reader as PMReader
from pmtiles.reader import MmapSource, all_tiles
from rasterio.coords import BoundingBox
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io.rasterio import Reader
import rasterio
from supermercado.burntiles import burn
import numpy as np

from .. import console, track

# nearest, box, bilinear, hamming, bicubic, lanczos
RESAMPLING_METHODS = {
    'nearest':  Resampling.NEAREST,
    'box':      Resampling.BOX,
    'bilinear': Resampling.BILINEAR,
    'hamming':  Resampling.HAMMING,
    'bicubic':  Resampling.BICUBIC,
    'lanczos':  Resampling.LANCZOS
}

def convert_rio_bbox(bbox: BoundingBox) -> LngLatBbox:
    return LngLatBbox(
        west=bbox.left,
        east=bbox.right,

        north=bbox.top,
        south=bbox.bottom
    )

def convert_bbox_to_geojson(bbox: LngLatBbox) -> Polygon:
    return Polygon(
        type='Polygon',
        coordinates=[[
            # (bbox.north, bbox.east),
            # (bbox.south, bbox.east),
            # (bbox.south, bbox.west),
            # (bbox.north, bbox.west),
            # (bbox.north, bbox.east)
            (bbox.east, bbox.north),
            (bbox.east, bbox.south),
            (bbox.west, bbox.south),
            (bbox.west, bbox.north),
            (bbox.east, bbox.north)
        ]]
    )



def get_bounding_tile(dataset: Reader) -> Tile:
    bbox = convert_rio_bbox(
        # the .info() converts to WGS84
        dataset.info().bounds
    )
    
    return bounding_tile(*bbox)



def bbox_overlap(bbox1: LngLatBbox, bbox2: LngLatBbox) -> bool:
    if bbox1.west > bbox2.east or bbox1.east < bbox2.west:
        # print('separating axis: east/west')
        return False

    if bbox1.north < bbox2.south or bbox1.south > bbox2.north:
        # print('separating axis: north/south')
        return False

    return True


def _calculate_tile_schedule_old(dataset: Reader, min_zoom: int = 1, max_zoom: int = 20) -> List[Tile]:
    bounding_tile = get_bounding_tile(dataset)

    bounding_bbox = convert_rio_bbox(
        dataset.info().bounds
    )


    # this gets split into five parts:
    # 0. add the bounding tile itself
    # 1. the parents of the bounding tile (bounding.z - min_zoom tiles)
    # 2. the children of the bounding tile (4^(max_zoom - bounding.z) tiles)
    # 3. filter out those tiles that aren't directly overlapping with the real bounding box
    # 4. sort by pmtiles tileid

    # step 0: add the bounding tile
    sched = [bounding_tile]

    # step 1: parents. we can skip this if the bounding tile's z is smaller
    # than the minimum requested zoom.
    for z in range(min_zoom, bounding_tile.z):
        print('step 1', z)
        sched.append(
            parent(bounding_tile, zoom=z)
        )

    # step 2: children.
    for z in range(bounding_tile.z+1, max_zoom+1):
        print('step 2', z)
        sched += children(bounding_tile, zoom=z)

    # step 3: filter out non-overlapping tiles
    sched = filter(
        lambda x: bbox_overlap(bounding_bbox, bounds(x)),
        sched
    )
    sched = list(sched)

    # print(f'filtered {orig_len} => {len(sched)}')

    # step 4: finally, to work around a bug in the pmtile writer, we need to
    # ensure that the tile schedule is ordered by the tileid
    sched.sort(key=lambda t:zxy_to_tileid(t.z,t.x,t.y))

    return sched

def _calculate_tile_schedule(dataset: Reader, min_zoom: int = 1, max_zoom: int = 20) -> List[Tile]:
    # step 1 : get the bounding polygon as a geojson feature
    bounding_bbox = convert_rio_bbox(
        dataset.info().bounds
    )
    bounding_poly = convert_bbox_to_geojson(bounding_bbox)
    bounding_poly = Feature(geometry=bounding_poly, type='Feature', properties=None)

    # next, use .burn(...) from supermercado as it's much faster than .children from mercantile
    sched_raws = []
    for z in range(min_zoom, max_zoom):
        sched_raw = burn([bounding_poly.dict()], zoom=z)
        sched_raws.append(sched_raw)


    # convert the tile matrix to a list of polys
    sched_raws = np.vstack(sched_raws)
    sched = []
    for i in range(sched_raws.shape[0]):
        row = sched_raws[i,:]
        sched.append(Tile(
            x=row[0],
            y=row[1],
            z=row[2]
        ))


    # step 3: filter out non-overlapping tiles
    sched = filter(
        lambda x: bbox_overlap(bounding_bbox, bounds(x)),
        sched
    )
    sched = list(sched)

    # print(f'filtered {orig_len} => {len(sched)}')

    # step 4: finally, to work around a bug in the pmtile writer, we need to
    # ensure that the tile schedule is ordered by the tileid
    sched.sort(key=lambda t:zxy_to_tileid(t.z,t.x,t.y))

    return sched

def create_png_tile(datasets: List[Reader], tile: Tile) -> bytes:
    tile_data = datasets[0].tile(tile.x, tile.y, tile.z)

    # create a Pillow image from the rearranged rio_tiler ImageData
    tile_img = Image.fromarray(
        tile_data.data_as_image()
    )

    tile_img.putalpha(
        Image.fromarray(tile_data.mask)
    )


    io = BytesIO()
    tile_img.save(io, format='PNG')

    return io.getvalue()

def create_numpy_tile(datasets: List[Reader], tile: Tile):
    tile_data = []
    for dataset in datasets:
        tile_data.append(
            dataset.tile(tile.x, tile.y, tile.z)
        )

    out_arr = np.stack( tile_data, axis=-1 )

    io = BytesIO()
    out_arr = np.save(io, out_arr)
    return io.getvalue()
    
def create_png_sar_tile(datasets: List[Reader], tile: Tile):
    assert len(datasets) == 2

    vv = datasets[0].tile(tile.x, tile.y, tile.z)
    vh = datasets[1].tile(tile.x, tile.y, tile.z)

    empty_channel = np.zeros_like(vv.array[0])
    out_arr = np.stack( [
        empty_channel,
        vv.array[0].copy(),
        vh.array[0].copy()
    ], axis=-1 ).astype(np.float64)


    # normalize to 256, clipping outliers
    out_arr /= 1000.0
    out_arr[out_arr > 1] = 1.0
    out_arr[out_arr < 0] = 0.0
    out_arr *= 256.0


    # create the output image with the data mask as the alpha channel
    tile_img = Image.fromarray(
        out_arr.astype(np.uint8)
    )
    tile_img.putalpha(
        Image.fromarray(vv.mask)
    )


    # write to png bytes
    io = BytesIO()
    tile_img.save(io, format='PNG')

    return io.getvalue()


def get_tile_schedule(infile: str, min_zoom=6, max_zoom=14, quiet=True) -> List[Tile]:

    with Reader(infile) as dataset:
        with console.status('generating tile schedule...') as status:
            sched = _calculate_tile_schedule(
                dataset,
                min_zoom=min_zoom,
                max_zoom=max_zoom
            )

        print(f'{len(sched)} tiles in schedule.')

        if not quiet:
            hist = {}
            for t in sched:
                if not t.z in hist:
                    hist[t.z] = 0
                hist[t.z] += 1

            hist = [(z, c) for z, c in hist.items()]
            hist.sort(key=lambda p: p[0])
            for z, c in hist:
                print(f'  zoom {z:2}: {c}')


        return sched

    
def convert_to_tiles(
        infiles: List[str],
        outfile: str,
        min_zoom: int = 6,
        max_zoom: int = 14,
        tile_type: TileType = TileType.PNG,
        tile_processor: Callable = create_png_tile
    ):
    '''

    :param infiles: List of input files to include in the raster. The first file sets the parameters for the pmtile archive info (e.g. bounds).
    :type infiles: List[str]
    :param outfile: File to write archive to
    :type outfile: str
    :param min_zoom: Minimum zoom level to construct. Default 6
    :type min_zoom: int
    :param max_zoom: Maximum zoom level to construct. Default 14
    :type max_zoom: int
    :param tile_type: The metadata of the tile type. This project uses Unknown to refer to Numpy tiles
    :type tile_type: pmtiles.tile.TileType
    :param tile_processor: Tile processing function. Takes a list of datasets and a tile, and returns a representation of that tile that should be saved.
    :type tile_processor: Callable[ [List[Reader]], [Any]]
    '''
    assert len(infiles) > 0

    datasets = []
    for infile in infiles:
        datasets.append(Reader(infile))


    to_e7 = lambda n: int(n * 10_000_000)

    bbox = convert_rio_bbox(
        # the .info() converts to WGS84
        datasets[0].info().bounds
    )

    pminfo = {
        'tile_type': tile_type, #TileType.PNG,
        'tile_compression': Compression.NONE,

        'min_zoom': min_zoom,
        'max_zoom': max_zoom,

        # these need to be switched because i'm pretty sure pmtiles has it backwards...
        'min_lon_e7': to_e7(bbox.west),
        'min_lat_e7': to_e7(bbox.south),
        'max_lon_e7': to_e7(bbox.east),
        'max_lat_e7': to_e7(bbox.north),

        'center_zoom': int(mean([min_zoom, max_zoom])),
        'center_lat_e7': to_e7(mean([bbox.east, bbox.west])),
        'center_lon_e7': to_e7(mean([bbox.north, bbox.south]))
    }


    sched = get_tile_schedule(
        infiles[0],
        min_zoom=min_zoom,
        max_zoom=max_zoom
    )

    skipped = 0

    with open(outfile, 'wb') as out_f:
        writer = PMWriter(out_f)

        for tile in track(sched, description='Creating tiles...', console=console):
            tileid = zxy_to_tileid(tile.z, tile.x, tile.y)
            # print(f'{tile} -> {tileid}')
            try:
                tile_bytes = tile_processor(datasets, tile)

                writer.write_tile(tileid, tile_bytes)
            except TileOutsideBounds:
                skipped += 1

        print(f'skipped {skipped} tiles')

        writer.finalize(
            pminfo,
            {
                'attribution': 'raster gen'
            }
        )

    osize = os.path.getsize(outfile)
    print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')

def convert_to_png_tiles(
        infile: str,
        outfile: str,
        min_zoom: int = 6,
        max_zoom: int = 14
    ):
    return convert_to_tiles(
        [infile],
        outfile,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        tile_type=TileType.PNG,
        tile_processor=create_png_tile
    )


def convert_to_png_sar_tiles(
        vv: str,
        vh: str,
        outfile: str,
        min_zoom: int = 6,
        max_zoom: int = 14
    ):
    return convert_to_tiles(
        [vv, vh],
        outfile,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        tile_type=TileType.PNG,
        tile_processor=create_png_sar_tile
    )


def get_tile_list(tilesets: List[str]) -> np.ndarray:
    tileid_list = set()
    with console.status('gathering metadata...'):
        for filename in tilesets:
            with open(filename, 'rb') as fp:
                source = MmapSource(fp)

                tiles = []
                for zxy, _ in all_tiles(source):
                    tileid = zxy_to_tileid(zxy[0], zxy[1], zxy[2])
                    tiles.append(tileid)

                # convert to a set to make my life a little easier
                tileid_list |= set(tiles)

    tileid_list = list(tileid_list)
    tileid_list.sort()

    return np.array(tileid_list)


def create_zoom_list(tile_list: np.ndarray, source_level: int, target_levels: List[int]) -> np.ndarray:
    source_tiles = []
    out_tiles = []
    out_ids = []

    for tile_id in track(tile_list, description='filtering tiles...'):
        z, x, y = tileid_to_zxy(tile_id)
        if z != source_level:
            continue

        source_tiles.append(Tile(z=z, x=x, y=y))

    for source_tile in track(source_tiles, description='generating list...'):
        for level in target_levels:
            out_tiles += children(source_tile, level=level)

    for tile in track(out_tiles, description='converting tiles...'):
        tile_id = zxy_to_tileid(tile.z, tile.x, tile.y)
        out_ids.append(tile_id)

    out_arr = np.unique( np.array(out_ids) )
    out_arr.sort()

    return out_arr


def merge_tilesets(tilesets, outfile):

    # helper function to get the metadata value
    def get_meta_key(metas, key):
        out = []
        for meta in metas:
            out.append(meta[key])
        return out


    all_meta = []
    tileid_list = {}
    with console.status('getting metadata...'):
        for filename in tilesets:
            with open(filename, 'rb') as fp:
                source = MmapSource(fp)
                reader = PMReader(source)
                all_meta.append(reader.header())

                tileid_list[filename] = []
                for zxy, _ in all_tiles(source):
                    tileid = zxy_to_tileid(zxy[0], zxy[1], zxy[2])
                    tileid_list[filename].append(tileid)

                # convert to a set to make my life a little easier
                tileid_list[filename] = set(tileid_list[filename])

    # pprint(all_meta)

    tile_list = []
    TileSource = namedtuple('TileSource', ['tileid', 'sources'])
    merges = {}

    schedule = []

    with console.status('determining merge schedule...'):
        for fn, lst in tileid_list.items():
            console.print(f'{fn}: {len(lst)} tiles')
            tile_list += list(lst)


        for fn, lst in tileid_list.items():
            for id in lst:
                if not id in merges:
                    merges[id] = set([fn])
                else:
                    merges[id].add(fn)

        copy_count = 0
        merge_count = 0
        for tileid, sources in merges.items():
            if len(sources) > 1:
                merge_count += 1
            else:
                copy_count  += 1
            # break

            schedule.append(TileSource(tileid=tileid, sources=sources))

        schedule.sort(key=lambda x: x.tileid)

    # calculate the meta block
    out_meta = {
        # todo: check if this holds across all tile types
        # we should check and transcode if we're not ready
        'tile_type': TileType.PNG,
        'tile_compression': Compression.NONE,

        # get from bounds
        'min_zoom': min(get_meta_key(all_meta, 'min_zoom')),
        'max_zoom': max(get_meta_key(all_meta, 'max_zoom')),

        'min_lon_e7': min(get_meta_key(all_meta, 'min_lon_e7')),
        'min_lat_e7': min(get_meta_key(all_meta, 'min_lat_e7')),
        'max_lon_e7': max(get_meta_key(all_meta, 'max_lon_e7')),
        'max_lat_e7': max(get_meta_key(all_meta, 'max_lat_e7')),

        # calculating on the fly
        'center_zoom': int(mean([
            min(get_meta_key(all_meta, 'min_zoom')),
            max(get_meta_key(all_meta, 'max_zoom'))
        ]))
    }
    # this stuff here needs to be in there but is calculated after the fact
    # to avoid some nasty duplication of work
    out_meta['center_lat_e7'] = int(mean([
        out_meta['min_lat_e7'],
        out_meta['max_lat_e7']
    ]))
    out_meta['center_lon_e7'] = int(mean([
        out_meta['min_lon_e7'],
        out_meta['max_lon_e7']
    ]))

    console.print(f'{merge_count} merges, {copy_count} copies (total: {merge_count + copy_count})')
    console.print(f' -> will save to [green]{outfile}[/]')

    # execute schedule

    # helper to merge png images
    def merge_pngs(pngs):
        # convert from byte files to Pillow images
        pngs = [ Image.open(BytesIO(p)) for p in pngs ]

        # we will paste all the images onto the first one in the list
        base = pngs[0]
        rest = pngs[1:]

        for img in rest:
            # the second img parameter is the alpha mask
            # so if there's transparency then that'll be included here
            base.paste(img, (0,0), img)
    
        # now, we just need to re-save these as png bytes
        io = BytesIO()
        base.save(io, format='PNG')
        return io.getvalue()

    with open(outfile, 'wb') as out_f:
        writer = PMWriter(out_f)

        # open all the readers
        readers = {}
        for filename in tilesets:
            with open(filename, 'rb') as fp:
                source = MmapSource(fp)
                readers[filename] = PMReader(source)

    
        # this part is easy
        for tile in track(schedule, description='Writing tiles...'):
            # annoyingly we need to do a little dance here
            z, x, y = tileid_to_zxy(tile.tileid)
            # and here
            sources = list(tile.sources)

            # if we just have one source, then this is easy: just copy from the reader to the writer
            if len(sources) == 1:
                png_bytes = readers[sources[0]].get(z, x, y)
                writer.write_tile(tile.tileid, png_bytes)
                continue

            # otherwise, we need to load all the bytes in for each source...
            png_bytes = []
            for src in sources:
                png_bytes.append(readers[src].get(z,x,y))

            # ... merge them ...
            merged = merge_pngs(png_bytes)

            # ... and then finally write them
            writer.write_tile(tile.tileid, merged)


        with console.status('finalizing bundle...'):
            writer.finalize(
                out_meta,
                {
                    'attribution': 'raster gen (merge)'
                }
            )
    osize = os.path.getsize(outfile)
    print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')


def combine_tiles_images(parent_tile: Tile, sub_tiles: List[Tuple[Tile, bytes]], resampling):
    tile_map = {}
    dims = None
    for tile, t_bytes in sub_tiles:
        tile_map[tile] = Image.open(BytesIO(t_bytes))
        dims = tile_map[tile].size

    # make a blank image
    if dims is None:
        return Image.new(mode='RGBA', size=(256, 256))

    width, height = dims
    output = Image.new(mode='RGBA', size=(width * 2, height * 2))

    positions = [ # top-left, top-right, bottom-right, bottom-left
        (0, 0),
        (width, 0),
        (width, height),
        (0, height)
    ]
    for child, pos in zip(children(parent_tile), positions):
        if not child in tile_map:
            continue

        output.paste(tile_map[child], pos)

    # resize the output to be the standard size
    output = output.resize(dims, resample=resampling)

    io = BytesIO()
    output.save(io, format='PNG')
    return io.getvalue()
    

def create_single_layer_downsample(tiles: List[Tuple[Tile, bytes]], resampling: Resampling):
    # create the map of which tiles go where
    next_map = {}

    for tile, t_bytes in tiles:
        parent_tile = parent(tile)

        if not parent_tile in next_map:
            next_map[parent_tile] = [(tile, t_bytes)]
        else:
            next_map[parent_tile].append((tile, t_bytes))

    # combine the tiles
    output = []
    for parent_tile, sub_tiles in track(next_map.items(), description='serializing tiles...'):
        output.append(
            (parent_tile, combine_tiles_images(parent_tile, sub_tiles, resampling))
        )

    return output
        

def create_downsamples(filename: str, outfile: str, source_level: Optional[int], final_level: int, resampling: Optional[Resampling] = Resampling.NEAREST):

    other_tiles: List[Tuple[Tile, bytes]] = []
    tile_list: List[Tuple[Tile, bytes]] = []
    header: Dict = {}

    # first, load in the entire dataset
    with open(filename, 'rb') as fp:
        source = MmapSource(fp)
        reader = PMReader(source)
        header = reader.header()

        if source_level is None:
            source_level = header['min_zoom']

        assert source_level > final_level

        for zxy, t_bytes in track(all_tiles(source), description='Reading dataset...'):
            z, x, y = zxy
            tile = Tile(z=z, x=x, y=y)

            # save the tiles at the source level to a separate list
            if z == source_level:
                tile_list.append( (tile, t_bytes) )
            else:
                other_tiles.append( (tile, t_bytes) )

    console.log(f'{len(tile_list)} tiles at source level {source_level}, {len(other_tiles)} others ({len(tile_list) + len(other_tiles)} total)')

    # start generating the tile list
    generated_tiles: Dict[int, List[Tuple[Tile, bytes]]] = {}
    generated_tiles[source_level] = tile_list
    for z in range(source_level-1, final_level-1, -1):
        console.log(f'processing level {z}...')
        generated_tiles[z] = create_single_layer_downsample( generated_tiles[z + 1], resampling)
        console.log(f'generated {len(generated_tiles[z])} tiles.')

    for z, tiles in generated_tiles.items():
        console.print(f'level {z:02}: {len(tiles)} tiles')


    # combine the tile lists into one full list
    # start with the non-munged tiles
    output_tiles = [(zxy_to_tileid(t.z, t.x, t.y), b) for (t, b) in other_tiles]

    # then combine the tiles layer-by-layer
    for _, tiles in generated_tiles.items():
        output_tiles += [(zxy_to_tileid(t.z, t.x, t.y), b) for (t, b) in tiles]

    # sort by tile_id
    output_tiles.sort(key=lambda x: x[0])

    # update the metadata
    header['min_zoom'] = final_level
    header['center_zoom'] = int(mean([
        header['min_zoom'],
        header['max_zoom']
    ]))

    with open(outfile, 'wb') as out_f:
        writer = PMWriter(out_f)

        # ... and then finally write them
        for tile_id, tile_bytes in output_tiles:
            writer.write_tile(tile_id, tile_bytes)


        with console.status('finalizing bundle...'):
            writer.finalize(
                header,
                {
                    'attribution': 'raster gen (downsample)'
                }
            )

    osize = os.path.getsize(outfile)
    console.print(f'Total bundle size: {(osize/(1024*1024)):.2f} MB')


def combine_tiffs(
        infile_1: str,
        infile_2: str,
        outfile: str
    ):
    

    with rasterio.open(infile_1) as src_1, rasterio.open(infile_2) as src_2:
        # first file metadata takes priority
        meta = src_1.meta.copy()

        # probably don't need this
        # meta.update(driver='GTiff', dtype='float32')

        with rasterio.open(outfile, 'w', **meta) as dst:
            band_1 = src_1.read(1)
            band_2 = src_2.read(1)
            
            combined_array = np.stack([
                band_1,
                band_2
            ])

            dst.write(combined_array)


if __name__ == '__main__':
    import cProfile

    with cProfile.Profile() as pr:
        get_tile_schedule(
            'test/landsat_2016_flood.tif',
            quiet=False,
            max_zoom=11
        )

        pr.dump_stats('trace.prof')
    # convert_to_tiles(
    #     'test/sentinel.jp2',
    #     'output.pmtile',
    #     max_zoom=12
    # )
