##Dependencies

1. numpy
2. matplotlib
3. scipy

The source code can be run under windows or linux with python 2.7+ and the libraries above.

##Directory structure
    .
    ├── asset
    │   ├── digitstest.txt  (test set, the last line is removed since its size is wrong)
    │   ├── digitstra.txt  (training set)
    │   ├── error-curve.png  (epoch-error plot)
    │   └── learning-curve.png  (data size-precision plot)
    ├── doc
    │   └── report.pdf   (the report)
    └── src
        ├── main.py  (generate the plots)
        ├── network.py  (for network building, training and classifying)
        └── util.py  (directory structure configurations)

##How to generate the results

Note: python scripts should be run under the `src` directory. All images will be placed under the `asset` directory.

1. Make sure the data set `digitstra.txt` and `digitstest.txt` is placed under `asset`
2. Run `python main.py` under `src`. The plots will be placed under `asset` named `error-curve.png` and `learning-curve.png`

##About

* [Github repository](https://github.com/joyeecheung/neural-network.git)
* Time: Jan. 2015
