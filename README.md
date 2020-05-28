# MoSeq2-Viz: Visualization toolbox for MoSeq2

Latest version number: `0.3.0`

## Features 

Below are the commands/functionality that moseq2-viz currently affords. 
They are accessible via CLI or Jupyter Notebook in [moseq2-app](https://github.com/dattalab/moseq2-app/tree/release). 
```bash
Usage: moseq2-viz [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.  [default: False]

Commands:
  add-group                       Change group name in index file given a...
  copy-h5-metadata-to-yaml        Copies metadata within an h5 file to a...
  make-crowd-movies               Writes movies of overlaid examples of the...
  plot-group-position-heatmaps    Plots position heatmaps for each group in...
  plot-scalar-summary             Plots a scalar summary of the index file...
  plot-syllable-durations         Plots syllable durations with different...
  plot-syllable-speeds            Plots syllable centroid speeds with...
  plot-transition-graph           Plots the transition graph depicting the...
  plot-usages                     Plots syllable usages with different...
  plot-verbose-position-heatmaps  Plots a position heatmap for each session...
  version                         Print version number
```

## Documentation

All documentation regarding moseq2-model can be found in the `Documentation.pdf` file in the root directory.

An HTML ReadTheDocs page can be generated using the `sphinx` package via running the `make html` command 
in the `docs/` directory.

To use this package, you must already have computed a `pca_scores.h5` files, a trained model: `model.p`, an index file
 `moseq2-index.yaml` listing all your analyzed sessions and paths to their extracted `.h5` files.  

 - The index file is generated when aggregating the results in [moseq2-extract](https://github.com/dattalab/moseq2-extract/tree/release) 
 - The pca_scores are generated via [moseq2-pca](https://github.com/dattalab/moseq2-pca/tree/release).
 - The model is generated via [moseq2-model](https://github.com/dattalab/moseq2-model/tree/release).
 
 
## Example Outputs

#### Crowd Movie Example:

<img src="https://drive.google.com/uc?export=view&id=1WB-H-3dqmFhEJbP9LCBAkkYldTmwpI7E" width=350 height=350>

#### Usage Plot Example:

<img src="https://drive.google.com/uc?export=view&id=1JiEdBL-sbi6DWw2jykrX4KUuH8RWKEFT">

#### Alternative Sorting Examples:

##### Usages by Mutated Behaviors (most mutated to least)

<img src="https://drive.google.com/uc?export=view&id=1VUACpHMTaNx1hqXtkkjCOJ6uQeo4ImhX">

##### Sorting Syllable Statistic in Descending Order (Speed shown)

<img src="https://drive.google.com/uc?export=view&id=14bJ7JZ9pSPgEtnJs6i7zfXW3GsS5IPAf">


#### Scalar Summary Example:

<img src="https://drive.google.com/uc?export=view&id=11XljpSbU_2Kx_3FTKvGQ00xz-rHwWZu2">

#### Position Heatmap Example:

<img src="https://drive.google.com/uc?export=view&id=1NR3EfhOx2JMTZeQdprVisIcHSHMv-0sb">

#### Transition Graph Example:

<img src="https://drive.google.com/uc?export=view&id=1j-ub8CfbHY5MKksL-PiwLhBLz-q2MiTQ" width=500 height=500>


## Contributing

If you would like to contribute, fork the repository and issue a pull request.  
