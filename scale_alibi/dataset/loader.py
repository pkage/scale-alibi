from io import BytesIO
import os
from typing import Any, List

from PIL import Image
import mercantile
from pmtiles.reader import Reader as PMReader
from pmtiles.reader import MmapSource, all_tiles
from pmtiles.tile   import TileType, tileid_to_zxy, zxy_to_tileid
import numpy as np
import torch
from torch.utils.data import Dataset


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


class RasterDataset(Dataset):
    raster_filename: str
    tile_ids: List[int]


class TileDataset(Dataset):
    tile_filename: str
    tile_reader: PMReader

    tile_type: TileType
    tile_ids: List[int]

    transform: Any


    def __init__(self, sample_tiles: str, transform=None):
        '''
        Initialize a new pmtile dataset.

        :param sample_tiles: tile archive (filename)
        :type sample_tiles: str
        :param transform: Transform to use for the samples
        :type transform: Any
        '''
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
        self.tile_type = self.sample_tiles_reader.header()['tile_type']

        # get the overlapping tile pairs by loading the tilesets
        # and using a set intersection
        tile_list = get_tile_list(self.tile_reader.get_bytes)
        self.tile_ids = list(tile_list)
        self.tile_ids.sort() # sort by ID to ensure reproducibility


    def __len__(self):
        '''
        Return length of valid tile list.

        :return: dataset length
        :rtype: int
        '''
        return len(self.tile_ids)


    def get_tile_from_index(self, index) -> mercantile.Tile:
        tile_id = self.valid_tile_ids[index]        # load the bytes for both sample and label
        z, x, y = tileid_to_zxy(tile_id)

        return mercantile.Tile(z=z, x=x, y=y)

    def __getitem__(self, index):
        # first, get the tileid
        tile_id = self.valid_tile_ids[index]        # load the bytes for both sample and label
        z, x, y = tileid_to_zxy(tile_id)

        # load the bytes of the images
        sample_bytes = self.sample_tiles_reader.get(z, x, y)
        label_bytes  = self.label_tiles_reader.get(z, x, y)

        # right now we're not handling unknown data
        sample_pair = {
            'sample': image_bytes_to_array(sample_bytes),
            'label':  image_bytes_to_array(label_bytes)
        }

        if self.transform:
            sample_pair = self.transform(sample_pair)

        return sample_pair

    # pickle handling
    def __getstate__(self):
        # for pickling, blank out the mmap handles as they don't work
        self.sample_tiles_reader = None
        self.label_tiles_reader = None
        return self.__dict__

    def __setstate__(self, unpickled_dict):
        self.__dict__ = unpickled_dict

        # for unpickling, reinitialize the mmap handles
        self.sample_tiles_reader = PMReader(
            MmapSource(open(self.sample_tiles_filename, 'rb'))
        )
        self.label_tiles_reader = PMReader(
            MmapSource(open(self.label_tiles_filename,  'rb'))
        )
        


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

