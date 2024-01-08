# Quality assessment algorithm evaluation

This repository currently contains "Error versus Discard Characteristic" (EDC) Python example/utility code, which can be used to evaluate quality assessment algorithms independent of the biometric modality.

- `edc.py`: Contains general code to compute EDC curve data, including "partial Area Under Curve" (pAUC) values. The only required Python package for this module is `numpy`.
- `example.py`: Contains a small CLI example that produces an EDC plot using the `plotly` package. The plot will be opened in the default browser. This example also includes plot utility functions.
- `example_data.json`: Default example input data used by `example.py`.

To quickly create a new Anaconda Python environment for `example.py`, run:

```
conda create --name edcexample python=3.9
conda activate edcexample
pip install -r requirements.txt
python example.py
```

There are various CLI options in the example, see `python example.py --help`.

A related "Face Image Quality Assessment Toolkit (fiqat)" with various additional features can be found here: <https://share.nbl.nislab.no/g03-03-sample-quality/face-image-quality-toolkit>
