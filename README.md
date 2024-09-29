# Scale-ALiBi implementation

This repository contains all of the code for running and training Scale-ALiBi, as well as generating the dataset and reproducing CROMA's results.

Lay of the land:

Folder | Purpose
--- | ---
`lambdas` | AWS Lambdas for managing the dataset/tiles/checkpoints
`notebooks` | Experimental notebooks
`scale_alibi` | Python package containing all the code, see below for usage
`site` | Scale-ALiBi website source
`slurm` | Template SLURM jobs for training the model

## Getting the code

### setup

You will need:

1. Python (3.11 or higher)
2. [Poetry](https://python-poetry.org)
3. (optional) [`poetry-jupyter-plugin`](https://github.com/pkage/poetry-jupyter-plugin)

```
$ git clone https://github.com/pkage/scale-alibi
$ cd scale-alibi
$ poetry install
```

if you'd like to use the notebooks and you have `poetry-jupyter-plugin`

```
$ poetry jupyter install
```


### training & inference



### creating a new dataset



## Citing

If you'd like to use this code, please cite:

```
@inproceedings{kage_scalealibi_2024,
  title = {{{Multi-modal}}, {{multi-scale}} {{representation learning}} for {{satellite imagery}} analysis just needs a good {{ALiBi}}}
  shorttitle = {{{Scale-ALiBi}}},
  author = {Kage, Patrick and Andreadis, Pavlos},
  year = {2024},
  month = oct,
  abstract = {
      Vision foundation models have been shown to be effective at processing
      satellite imagery into representations fit for downstream tasks, however,
      creating models which operate over multiple spatial resolutions and modes
      is challenging. This paper presents Scale-ALiBi, a linear bias transformer
      attention mechanism with a spatial encoding bias to relationships between
      image patches at different ground sample distance scales. We provide an
      implementation of Scale-ALiBi over a dataset of aligned high- and
      low-resolution optical and low-resolution SAR satellite imagery data using
      a triple-contrastive and reconstructive architecture, show an improvement
      on the GEO-Bench benchmark, and release the newly curated dataset publicly.
  },
  copyright = {MIT License},
  howpublished = {The University of Edinburgh}
}
```

