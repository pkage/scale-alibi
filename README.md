# scale-alibi implementation

`// todo: write readme`

# notes

## imagery acesss

for sentinel-* access:

```
$ export STAC_API_URL=https://earth-search.aws.element84.com/v1
$ stac-client search ${STAC_API_URL} -c sentinel-2-l2a --bbox -72.5 40.5 -72 41 --datetime 2020-01-01/2020-01-31 --matched
```

will need to impl. parser and downloader, but that seems doable. i don't need this to work *consistently*, just enough to get some rasters down.


download from aws opendata using a req. pays bucket ([sentinel-1](https://aws.amazon.com/marketplace/pp/prodview-uxrsbvhd35ifw?sr=0-18&ref_=beagle&applicationId=AWSMPContessa#resources), [sentinel-2](https://aws.amazon.com/marketplace/pp/prodview-2ostsvrguftb2?sr=0-20&ref_=beagle&applicationId=AWSMPContessa#usage))


i ... might be able to do this with `jq` and tha `aws` cli, actually

for example:

NW atlanta (z/x/y): `9/135/204`

Download the latest TCI image:

```sh
# create the geojson of the search area (xyz)
echo "[135,204,9]" | mercantile shapes | jq .bbox -c | sed 's/\[//;s/\]//;s/,/ /g'

# create the search
stac-client search https://earth-search.aws.element84.com/v1 \
    -c sentinel-2-l2a \
    --bbox -85.078 33.724 -84.375 34.307 \
    --query "eo:cloud_cover<10" \ 
    --datetime 2024-08-01/2024-08-12 | jq . > search.json


# for s3 requester pays
# aws s3 cp --request-payer requester  `jq '.features[0].assets."visual-jp2".href' -r < search.json` .

curl `jq '.features[0].assets.visual.href' -r < search.json` -o visual.tif

```

similarly, for sentinel-1:

```sh
# create the geojson of the search area (xyz)
echo "[135,204,9]" | mercantile shapes | jq .bbox -c | sed 's/\[//;s/\]//;s/,/ /g'

# execute the search, this time without cloud cover filter (not necessary for SAR)
stac-client search https://earth-search.aws.element84.com/v1 \
    -c sentinel-1-grd \
    --bbox -85.078 33.724 -84.375 34.307 \
    --datetime 2024-08-01/2024-08-12 | jq . > search-s1.json

# and download
aws s3 cp  `jq '.features[0].assets.vv.href' -r < search-s1.json` vv.tif --request-payer requester
aws s3 cp  `jq '.features[0].assets.vh.href' -r < search-s1.json` vh.tif --request-payer requester
```

```sh

```
