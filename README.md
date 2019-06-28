# Benchmark signature features on UCR/UEA datasets

This repo provides the code to run simple benchmarks
for signature features on the [UCR/UEA Time Series
Classification repository](http://www.timeseriesclassification.com/). (Work in progress)


## Results

Some of our benchmark results are uploaded in this Github
repository under `results/`. In the [`result_analysis.ipynb`](./result_analysis.ipynb) Jupyter
notebook, you can view a brief analysis of these results.


## Installation

This project uses the Python packages `iisignature`, `sklearn`, `numpy`, `pandas`, `sktime`.
Furthermore, `luigi` is used for orchestration of different tasks (slight overkill currently).

First clone the repo:
```
git clone https://github.com/zhy0/sig-tsc.git
```

Create a Python 3 virtual env and install the dependencies:
``` bash
cd sig-tsc
python3 -m venv env
source env/bin/activate
# install numpy and cython separately, this is due to an issue with installing sktime
pip install numpy
pip install cython
# install rest of the dependencies
pip install -r requirements.txt
```
This will create the folder `pipeline` under the
`sig-tsc` folder which contains the results of the classification.


## Downloading the datasets at the right location

Enter the `sig-tsc` folder and run the following commands:
``` bash
# UCR datasets
wget http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip
unzip Univariate2018_arff.zip

# UEA datasets (multivariate)
wget http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip
unzip Multivariate2018_arff.zip
```

Unzipping will create two folders, `NewTSCProblems` and `MultivariateTSCProblems`.
In `tasks.py`, the location of these two folders can be changed. By default
they're assumed to be in the `sig-tsc` folder.


## Usage

To run a benchmark of a single dataset, you can use the `luigi` command line:
``` bash
python -m luigi --module tasks RunUnivariate --levels '[2,3,4]' --dataset ECG200 --model-type sklearn.svm.LinearSVC --sig-type sig --local-scheduler
```
See `tasks.py` for the individual parameters.
(The `local-scheduler` flag tells Luigi to use the local scheduler.)

Running benchmarks on all datasets can done using
``` bash
python tasks.py
```
You can edit the contents of `main()` to change the behavior.
