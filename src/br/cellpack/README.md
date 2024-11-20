# cellPACK

Methods for creating synthetic data using cellPACK.

## Installation

To install cellPACK, follow the instructions in the [cellPACK documentation](https://github.com/mesoscope/cellpack).

Install libraries required for generating the synthetic data (it is recommended to use a virtual environment):

```bash
pip install fire
pip install quilt3
```

## Usage

1. Get the reference nuclear shapes:

```bash
python get_reference_nuclear_shapes.py
```

2. Generate the synthetic data:

```bash
python generate_synthetic_data.py
```

Additional options can be specified through the command line. Run `python generate_synthetic_data.py --help` for more information.

The generated synthetic data will be saved in `data/packings`
