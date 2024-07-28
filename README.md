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
