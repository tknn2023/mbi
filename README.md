# Multi-level Block Indexing

## Requirements

- Python 3.7+
- cargo 1.63.0+
- maturin 0.14.10+

## Compile

```bash
maturin build -r
# or 
maturin develop -r
```

## Usage

In Python, import the package by

```python
import knnrs
```

```py
knnrs.TknnMBI.build_graph<'py>(data, label, n_neighbors, max_candidates, max_iters, rpdiv_iter, rpdiv_size_limit, tf_eps, dist, k, tknn_leaf_size)
```

Build a graph from data.
`data`: `np.ndarray<f32>[n_data, dim]` Vectors to be indexed.  
`label`: `np.ndarray<i64>[n_data]` Labels or timestamps of vectors.  
`n_neighbors`: `int` The number of neighbors per vertex when creating a KNNGraph using the NNDescent algorithm. A range of 20 to 200 is recommended.
`max_candidates`: `int` An attribute used when building a graph with the NNDescent algorithm. During neighbor updates, the number of neighbors' neighbors to be considered in each iteration at each vertex is limited to the number of max_candidates. A value 1 to 2 times that of n_neighbors is recommended.
`max_iters`: `int` The max number of iterations for neighbor updates in the NNDescent algorithm.
`rpdiv_iter`: `int` A property used when building a graph with the NNDescent algorithm. A value of 15 is recommended.
`rpdiv_size_limit`: `int` A property used when building a graph with the NNDescent algorithm. A value of 1024 is recommended.
`tf_eps`: `float` A property used when building a graph. A value of 0.6 is recommended.
`dist`: `str` The distance metric to use. One of `['euclidean', 'cosine', 'normed_cosine']`.
`k`: `int` The number of neighbors to be searched.
`tknn_leaf_size`: `int` The leaf size of the tree used in the MBI algorithm. It is recommended to use about 5 to 10% of the data size.

`knnrs.TknnMBI.single_query(self, q, start_label, end_label)`  
Search for k nearest neighbors from a single query point on the graph.
`q`: `np.ndarray<f32>[dim]` A query point to be searched.
`start_label`: `i64` The start label of the search interval.
`end_label`: `i64` The end label of the search interval.

## Dataset

| Dataset | Description | Dimension | #Vectors | Source |
| --- | --- | --- | --- | --- |
| COMS | Weather satellite images captured by the Communication, Ocean and Meteorological Satellite (COMS), which is the first geostationary multipurpose satellite of South Korea. | 128 (Need to be preprocessed.) | 291,380 | [data.go.kr](https://www.data.go.kr/data/15058167/openapi.do) |
| MovieLens | Set of movies, each represented as a 32-dimensional vector with its release year as the timestamp | 32 (Need to be preprocessed.) | 57,771 | [ics.uci.edu](https://grouplens.org/datasets/movielens/) |
| Glove-25 | Global vectors for word representation in web dataset | 25 | 1,193,514 | [nlp.stanford.edu](https://nlp.stanford.edu/projects/glove/) |
