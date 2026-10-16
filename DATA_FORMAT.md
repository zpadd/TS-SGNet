# Input formats — synthetic examples only

The model expects a 200 × 200 grid and seven feature columns: `TOC`, `RO`, `ST`, `SRH`, `RE`, `GT`, `TE`.

### Wells: `example_data/蠡县探井连续数据集.csv`

Columns: `index,X,Y,TOC,RO,ST,SRH,RE,GT,TE,CLASS`.

`index` uses fabricated IDs `Well-1` through `Well-32`. `CLASS` is a binary 0/1 label. `X` and `Y` are synthetic grid indices, **not** geographic coordinates. The seven attributes are fabricated binary values.

### Grid: `example_data/蠡县网格连续数据集.csv`

Columns: `index,X,Y,TOC,RO,ST,SRH,RE,GT,TE`.

Contains 40,000 synthetic rows on a 200 × 200 lattice, indexed `Cell-1` through `Cell-40000`. All feature values are artificial 0/1 placeholders.

### Work boundary: `example_data/蠡县新工区边界.txt`

A closed artificial polygon in `x,y` text format.

### Reserve outline: `example_data/蠡县新工区储量.bln`

A fabricated BLN-style polygon: the header specifies vertex count and flags, followed by coordinate pairs.

### Privacy and reproducibility

All names, coordinate values, attributes, labels and geometries were generated for this repository, independently of the restricted study data. These files neither preserve the original sample distribution nor reveal actual locations. They are suitable for inspecting schemas, not for reproducing the article's evaluation metrics. The restricted original data cannot be supplied, even upon request.
