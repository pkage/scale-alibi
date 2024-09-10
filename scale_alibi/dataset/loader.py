from abc import abstractmethod
from collections import namedtuple
from .. import console
from io import BytesIO
import os
from re import L
from typing import Any, List

from PIL import Image
import mercantile
from pmtiles.reader import Reader as PMReader
from pmtiles.reader import MmapSource, all_tiles
from pmtiles.tile   import TileType, tileid_to_zxy, zxy_to_tileid
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io.rasterio import Reader
from rio_tiler.models import ImageData
from .tile import get_tile_schedule

import numpy as np
import torch
from torch.utils.data import Dataset

MultimodalSample = namedtuple('MultimodalSample', ['radar', 'lores', 'hires', 'tile_id'])
LoresMultimodalSample = namedtuple('LoresMultimodalSample', ['radar', 'lores', 'tile_id'])

# --- HELPERS ---

def get_tile_list(source) -> List[int]:
    # only hang on to the tile_ids
    tile_list = [zxy_to_tileid(z,x,y) for (z,x,y), _ in all_tiles(source)]

    return tile_list


def image_bytes_to_array(img_bytes: bytes) -> np.ndarray:
    img = Image.open(
        BytesIO(img_bytes)
    )

    return np.array(img, dtype=np.uint8)


def numpy_bytes_to_array(np_bytes: bytes) -> np.ndarray:
    arr = np.load(
        BytesIO(np_bytes)
    )

    return arr

# --- LOADERS ---

class TileIdDataset(Dataset):
    tile_ids: List[int]
    transform: Any

    def __init__(self, transform=None):
        self.transform = transform
        self.tile_ids = []


    def __len__(self):
        return len(self.tile_ids)

    def get_tile_from_index(self, index) -> mercantile.Tile:
        tile_id = self.tile_ids[index]
        z, x, y = tileid_to_zxy(tile_id)
        return mercantile.Tile(z=z, x=x, y=y)

    def has_tile_id(self, tile_id):
        return tile_id in self.tile_ids

    def __getitem__(self, index) -> Any:
        # first, get the tileid
        tile_id = self.tile_ids[index]
        return self.get_by_tile_id(tile_id)

    def _apply_transform(self, data):
        if self.transform is not None:
            return self.transform(data)

        return data

    @abstractmethod
    def get_by_tile_id(self, tile_id) -> Any:
        # should call self._apply_transform(...) somewhere
        ...



class RasterDataset(TileIdDataset):
    raster_filename: str
    tile_reader: Reader

    min_zoom: int
    max_zoom: int

    def __init__(self, raster_filename: str, transform=None, zoom_level=16):
        super().__init__(transform=transform)
        # load tile ids
        tile_sched = get_tile_schedule(raster_filename, min_zoom=zoom_level, max_zoom=zoom_level+1)
        self.tile_ids = [ zxy_to_tileid(t.z, t.x, t.y) for t in tile_sched ]
        self.tile_ids.sort() # sort by ID to ensure reproducibility

        self.raster_filename = raster_filename
        self.tile_reader = Reader(self.raster_filename)

        self.transform = transform 



    def get_by_tile_id(self, tile_id) -> ImageData:
        z, x, y = tileid_to_zxy(tile_id)

        # load the bytes of the images
        tile = self.tile_reader.tile(z, x, y)

        return self._apply_transform(tile)
        

    # pickle handling
    def __getstate__(self):
        # for pickling, blank out the mmap handles as they don't work
        self.tile_reader = None
        return self.__dict__

    def __setstate__(self, unpickled_dict):
        self.__dict__ = unpickled_dict

        # for unpickling, reinitialize the mmap handle
        self.tile_reader = PMReader(
            MmapSource(open(self.tile_filename, 'rb'))
        )


class PMTileDataset(TileIdDataset):
    tile_filename: str
    tile_reader: PMReader

    tile_type: TileType

    transform: Any


    def __init__(self, sample_tiles: str, transform=None):
        '''
        Initialize a new pmtile dataset.

        :param sample_tiles: tile archive (filename)
        :type sample_tiles: str
        :param transform: Transform to use for the samples
        :type transform: Any
        '''
        super().__init__(transform=transform)
        assert os.path.exists(sample_tiles)

        # save the transform
        self.transform = transform

        # save the filenames
        self.tile_filename = sample_tiles

        # finally, set the loader mmap sources
        self.tile_reader = PMReader(
            MmapSource(open(self.tile_filename, 'rb'))
        )

        # load the tile type information
        self.tile_type = self.tile_reader.header()['tile_type']

        # get the overlapping tile pairs by loading the tilesets
        # and using a set intersection
        tile_list = get_tile_list(self.tile_reader.get_bytes)
        self.tile_ids = list(tile_list)
        self.tile_ids.sort() # sort by ID to ensure reproducibility


    def get_by_tile_id(self, tile_id) -> np.ndarray:
        z, x, y = tileid_to_zxy(tile_id)

        # load the bytes of the images
        tile_bytes = self.tile_reader.get(z, x, y)

        # right now we're not handling unknown data
        tile_arr = image_bytes_to_array(tile_bytes)

        return self._apply_transform(tile_arr)


    def __getitem__(self, index) -> np.ndarray:
        # first, get the tileid
        tile_id = self.tile_ids[index]
        return self.get_by_tile_id(tile_id)


    # pickle handling
    def __getstate__(self):
        # for pickling, blank out the mmap handles as they don't work
        self.tile_reader = None
        return self.__dict__

    def __setstate__(self, unpickled_dict):
        self.__dict__ = unpickled_dict

        # for unpickling, reinitialize the mmap handle
        self.tile_reader = PMReader(
            MmapSource(open(self.tile_filename, 'rb'))
        )


# --- MULTI-MODAL ACCESSORS ---

class TileUnionDataset(TileIdDataset):
    '''
    Represent a dataset as composed of union of smaller raster resources
    '''
    datasets: List[TileIdDataset] # can probably be smarter about this
    transform: Any
    tile_ids: List[int]

    def __init__(self, transform=None, datasets: List[TileIdDataset] = []):
        self.transform = transform
        self.datasets = datasets
        self.recalculate_tile_list()

    def recalculate_tile_list(self):
        # calculate all tiles available in dataset
        valid_tiles = set(self.datasets[0].tile_ids)
        for dset in self.datasets[1:]:
            valid_tiles |= set(dset.tile_ids)

        # and using a set intersection
        self.tile_ids = list(valid_tiles)
        self.tile_ids.sort() # sort by ID to ensure reproducibility


    def add_dataset(self, dataset):
        self.datasets.append(dataset)
        self.recalculate_tile_list()


    def get_by_tile_id(self, tile_id):
        tile_arr = []

        for dset in self.datasets:
            if dset.has_tile_id(tile_id):
                tile_arr.append(dset.get_by_tile_id(tile_id))


        return self._apply_transform(tile_arr)


class LoresMultimodalDataset(TileIdDataset):
    radar_datasets: TileIdDataset
    lores_datasets: TileIdDataset

    def __init__(
            self,
            radar_datasets: TileIdDataset,
            lores_datasets: TileIdDataset,
            transform=None
        ):
        super().__init__(transform=transform)
        self.radar_datasets = radar_datasets
        self.lores_datasets = lores_datasets

        self.recalculate_tile_list()


    
    def recalculate_tile_list(self):
        # calculate all tiles available in dataset
        valid_tiles = set(self.radar_datasets.tile_ids) & set(self.lores_datasets.tile_ids)

        # and using a set intersection
        self.tile_ids = list(valid_tiles)
        self.tile_ids.sort() # sort by ID to ensure reproducibility


    def get_by_tile_id(self, tile_id) -> LoresMultimodalSample:
        sample = LoresMultimodalSample(
            radar=self.radar_datasets.get_by_tile_id(tile_id),
            lores=self.lores_datasets.get_by_tile_id(tile_id),
            tile_id=tile_id
        )

        return self._apply_transform(sample)


class MultimodalDataset(TileIdDataset):
    radar_datasets: TileIdDataset
    hires_datasets: TileIdDataset
    lores_datasets: TileIdDataset

    def __init__(
            self,
            radar_datasets: TileIdDataset,
            lores_datasets: TileIdDataset,
            hires_datasets: TileIdDataset,
            transform=None
        ):
        super().__init__(transform=transform)
        self.radar_datasets = radar_datasets
        self.lores_datasets = lores_datasets
        self.hires_datasets = hires_datasets

        self.recalculate_tile_list()


    def recalculate_tile_list(self):
        # calculate all tiles available in dataset
        valid_tiles = set(self.radar_datasets.tile_ids) & set(self.lores_datasets.tile_ids) & set(self.hires_datasets.tile_ids)

        # and using a set intersection
        self.tile_ids = list(valid_tiles)
        self.tile_ids.sort() # sort by ID to ensure reproducibility


    def get_by_tile_id(self, tile_id) -> MultimodalSample:
        sample = MultimodalSample(
            radar=self.radar_datasets.get_by_tile_id(tile_id),
            lores=self.lores_datasets.get_by_tile_id(tile_id),
            hires=self.hires_datasets.get_by_tile_id(tile_id),
            tile_id=tile_id
        )
    
        return self._apply_transform(sample)


# --- TRANSFORMS ---


class RandomFlip:
    '''Randomly flips the sample/labels horizontally and/or vertically'''

    def __call__(self, sample_pair):
        sample = sample_pair['sample']
        label = sample_pair['label']

        flip_h, flip_v = torch.randint(0,2,(2,))

        if bool(flip_h):
            sample = np.flip(sample, axis=0)
            label  = np.flip(label,  axis=0)

        if bool(flip_v):
            sample = np.flip(sample, axis=1)
            label  = np.flip(label,  axis=1)

        # remove negative strides
        if bool(flip_v) or bool(flip_h):
            sample = sample.copy()
            label = label.copy()

        return {
            'sample': sample,
            'label':  label
        }



class RescaleImageData:
    '''Rescale ImageData to be between 0-1'''

    def __init__(self, data_min=None, data_max=None):
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, image_data: ImageData):
        if type(image_data) is not ImageData:
            console.log(f'[red]{image_data} is not ImageData, this will probably break')

        stats = image_data.statistics(categorical=True)['b1']
        
        if self.data_min is None:
            data_min = stats.min
        else:
            data_min = self.data_min

        if self.data_max is None:
            data_max = stats.max
        else:
            data_max = self.data_max

        image_data.rescale(in_range=((data_min, data_max),),)

        return image_data

class ImageDataToNumpy:
    '''Convert ImageData to numpy, in (row, col, band) order'''

    def __call__(self, img: ImageData):
        if type(img) is not ImageData:
            console.log(f'[red]{img} is not ImageData, this will probably break')

        return img.data_as_image()


class ToTensor:
    '''Convert ndarrays in sample pair to Tensors.'''

    def __call__(self, sample_pair):
        sample = sample_pair['sample']
        label = sample_pair['label']


        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        sample = sample.transpose((2, 0, 1))
        label  = label.transpose((2, 0, 1))
        label = label[0,:,:]

        # clamp to 1.0
        label[label > 1.0] = 1.0

        return {
            'sample': torch.from_numpy(sample),
            'label':  torch.from_numpy(label)
        }

