import pandas as pd
import numpy as np
import pathlib
import luigi
import timeit
import classifiers
from model import SigModel, LogSigModel
from sktime.utils.load_data import load_from_arff_to_dataframe
from xgboost import XGBClassifier

import sklearn
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import pickle

DATA_DIR = pathlib.Path('NewTSCProblems')
MULT_DATA_DIR = pathlib.Path('MultivariateTSCProblems')
PIPELINE_DIR = pathlib.Path('pipeline') # directory to store results

TRANSFORMS = {
    'leadlag': SigModel.lead_lag,
    'timejoined': SigModel.time_joined,
    'timeindexed': lambda X: [(t,x) for t,x in enumerate(X)]
}

DATASETS = [
    "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY",
    "AllGestureWiimoteZ", "ArrowHead", "Beef", "BeetleFly", "BirdChicken",
    "BME", "Car", "CBF", "Chinatown", "ChlorineConcentration",
    "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY",
    "CricketZ", "Crop", "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame",
    "DodgerLoopWeekend", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays",
    "ElectricDevices", "EOGHorizontalSignal", "EOGVerticalSignal",
    "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords",
    "Fish", "FordA", "FordB", "FreezerRegularTrain", "FreezerSmallTrain",
    "Fungi", "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint", "GunPointAgeSpan",
    "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham",
    "HandOutlines", "Haptics", "Herring", "HouseTwenty", "InlineSkate",
    "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound",
    "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2",
    "Lightning7", "Mallat", "Meat", "MedicalImages", "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW", "MixedShapesRegularTrain", "MixedShapesSmallTrain",
    "MoteStrain", "NonInvasiveFatalECGThorax1",
    "NonInvasiveFatalECGThorax2", "OliveOil", "OSULeaf",
    "PhalangesOutlinesCorrect", "Phoneme", "PickupGestureWiimoteZ",
    "PigAirwayPressure", "PigArtPressure", "PigCVP", "PLAID", "Plane",
    "PowerCons", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
    "RefrigerationDevices", "Rock", "ScreenType", "SemgHandGenderCh2",
    "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShakeGestureWiimoteZ",
    "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
    "TwoPatterns", "UCR_archive_2018_to_release", "UMD",
    "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
]


def load_data(dataset, datadir=DATA_DIR):
    """
    Load UCR training and testing data from ARFF file and return each as numpy arrays.
    """
    X_train, y_train = load_from_arff_to_dataframe(datadir/dataset/f"{dataset}_TRAIN.arff")
    X_test, y_test = load_from_arff_to_dataframe(datadir/dataset/f"{dataset}_TEST.arff")

    def parse_to_np(data):
        return np.array([x for x in data['dim_0']])

    X_train = parse_to_np(X_train)
    X_test = parse_to_np(X_test)

    return (X_train, y_train, X_test, y_test)


def load_mult_data(dataset, datadir=MULT_DATA_DIR):
    """
    Load UES data from ARFF file and return as 3D numpy arrays.
    """
    X_train, y_train = load_from_arff_to_dataframe(datadir/dataset/f"{dataset}_TRAIN.arff")
    X_test, y_test = load_from_arff_to_dataframe(datadir/dataset/f"{dataset}_TEST.arff")

    def parse_to_np(data):
        return np.array([pd.concat(row.values, axis=1).values
                         for i, row in data.iterrows()])

    X_train = parse_to_np(X_train)
    X_test = parse_to_np(X_test)

    return (X_train, y_train, X_test, y_test)


def load_results(filename, results_dir='./results/ucr', datasets=DATASETS):
    """
    Load all results stored as a .pkl file.
    """
    r = []
    missing = []
    names = []
    for d in datasets:
        try:
            data = pd.read_pickle(f'{results_dir}/{d}/{filename}')
            names.append(d)
            r.append(data)
        except FileNotFoundError:
            missing.append(d)
    return (pd.concat(r, keys=names), missing)


class PickleTask(luigi.Task):
    """
    Task class to save Pandas dataframe as pickle. To be inherited.
    """
    def load(self):
        return pd.read_pickle(self.input())

    def dump(self, df):
        self.output().makedirs()
        df.to_pickle(self.output().path, compression=None)


class PreprocessMultivariate(luigi.Task):
    """
    Task to turn ARFF file data into numpy format.

    We use a separate task for this step because this can take a long
    time for large datasets.
    """
    dataset = luigi.Parameter()

    def output(self):
        fname = f"mult_{self.dataset}.npz"
        return luigi.LocalTarget(PIPELINE_DIR/'prep'/fname)

    def run(self):
        X_train, y_train, X_test, y_test = load_mult_data(self.dataset)
        out = self.output()
        out.makedirs()
        np.savez(out.path, X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)


class RunMultivariate(PickleTask):
    """
    Run all benchmarks for multivariate UEA datasets.

    Params:
        dataset: (str) name of the dataset, e.g., ECG200
        levels: list of truncation levels, e.g., [2,3,4]
        sig_type: 'sig' or 'logsig', type of signature features to use
    """
    dataset = luigi.Parameter()
    levels = luigi.ListParameter()
    sig_type = luigi.Parameter(default='sig')

    def requires(self):
        return PreprocessMultivariate(dataset=self.dataset)

    def output(self):
        levels_name = '_'.join(map(str, self.levels))
        filename = f"{self.sig_type}_{levels_name}.pkl"
        return luigi.LocalTarget(PIPELINE_DIR/'mult'/self.dataset/filename)

    def run(self):
        data = np.load(self.input().path, allow_pickle=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test  = data['X_test']
        y_test  = data['y_test']

        model = LogisticRegression
        r = []
        for level in self.levels:
            if self.sig_type == 'sig':
                m = SigModel(model, level=level)
            elif self.sig_type == 'logsig':
                m = LogSigModel(model, dim=X_train.shape[2], level=level)
            # start timing
            start = timeit.default_timer()
            m.train(X_train, y_train)
            elapsed = timeit.default_timer() - start
            # end timing
            r.append([m.score(X_test, y_test), elapsed])
        result = pd.DataFrame(r, columns=['Score', 'Elapsed'], index=self.levels)
        self.dump(result)


class RunUnivariate(PickleTask):
    """
    Run all benchmarks for univariate UCR datasets.

    Params:
        dataset: (str) name of the dataset, e.g., ECG200
        levels: (list) list of truncation levels, e.g., [2,3,4]
        sig_type: (str) 'sig' or 'logsig', type of signature features to use
        model_type: the name of the sklearn classification algorithm to use
    """
    dataset = luigi.Parameter()
    levels = luigi.ListParameter()
    sig_type = luigi.Parameter(default='sig')
    model_type = luigi.Parameter(default='LogisticRegression')
    model_args = luigi.DictParameter(default={})

    def output(self):
        levels_name = '_'.join(map(str, self.levels))
        filename = f"{self.sig_type}_{self.model_type}_{levels_name}.pkl"
        return luigi.LocalTarget(PIPELINE_DIR/self.dataset/filename)

    def run(self):
        X_train, y_train, X_test, y_test = load_data(self.dataset)
        model = eval(self.model_type) # this is unsafe
        #model = LinearSVC
        #model = LogisticRegression
        #model = KNeighborsClassifier

        scores = []
        times = []
        for t_name, transform in TRANSFORMS.items():
            r = []
            t = []
            for level in self.levels:
                if self.sig_type == 'sig':
                    m = SigModel(model, transform=transform, level=level, **self.model_args)
                elif self.sig_type == 'logsig':
                    m = LogSigModel(model, dim=2, transform=transform, level=level, **self.model_args)
                # start timing
                start = timeit.default_timer()
                m.train(X_train, y_train)
                elapsed = timeit.default_timer() - start
                # end timing
                r.append(m.score(X_test, y_test))
                t.append(elapsed)
            scores.append(r)
            times.append(t)

        # create two dataframes containing scores and training time
        scores = pd.DataFrame(scores, columns=self.levels,
                              index=[t for t in TRANSFORMS]).T
        times = pd.DataFrame(times, columns=self.levels,
                              index=[f'{t}_elapsed' for t in TRANSFORMS]).T

        self.dump(pd.concat([scores,times], axis=1))


class RunVotingEnsemble(PickleTask):
    dataset = luigi.Parameter()
    levels = luigi.ListParameter()
    sig_type = luigi.Parameter(default='sig')
    #clf_type = luigi.Parameter(default='LogisticRegression')
    #clf_args = luigi.DictParameter(default={})

    def output(self):
        levels_name = '_'.join(map(str, self.levels))
        filename = f"{self.sig_type}_flatcote_{levels_name}.pkl"
        return luigi.LocalTarget(PIPELINE_DIR/self.dataset/filename)

    def run(self):
        X_train, y_train, X_test, y_test = load_data(self.dataset)
        logit = LogisticRegression(random_state=42)

        r = []
        for level in self.levels:
            m = classifiers.create_vote_clf(logit, level=level)
            # start timing
            start = timeit.default_timer()
            m.fit(X_train, y_train)
            elapsed = timeit.default_timer() - start
            # end timing
            r.append([m.score(X_test, y_test), elapsed])

        self.dump(pd.DataFrame(r, columns=["Score", "Elapsed"], index=self.levels))


class RunFeatureUnion(PickleTask):
    dataset = luigi.Parameter()
    levels = luigi.ListParameter()
    sig_type = luigi.Parameter(default='logsig')

    def output(self):
        levels_name = '_'.join(map(str, self.levels))
        filename = f"{self.sig_type}_concat_{levels_name}.pkl"
        return luigi.LocalTarget(PIPELINE_DIR/self.dataset/filename)

    def run(self):
        X_train, y_train, X_test, y_test = load_data(self.dataset)
        logit = LogisticRegression(random_state=42)

        r = []
        for level in self.levels:
            m = classifiers.create_concatenator(logit, sig_type=self.sig_type, level=level)
            # start timing
            start = timeit.default_timer()
            m.fit(X_train, y_train)
            elapsed = timeit.default_timer() - start
            # end timing
            r.append([m.score(X_test, y_test), elapsed])

        self.dump(pd.DataFrame(r, columns=["Score", "Elapsed"], index=self.levels))


class RunLogit(PickleTask):
    dataset = luigi.Parameter()

    def output(self):
        filename = "logit.pkl"
        return luigi.LocalTarget(PIPELINE_DIR/self.dataset/filename)

    def run(self):
        X_train, y_train, X_test, y_test = load_data(self.dataset)
        logit = LogisticRegression(random_state=42)
        # start timing
        start = timeit.default_timer()
        logit.fit(X_train, y_train)
        elapsed = timeit.default_timer() - start

        score = logit.score(X_test, y_test)
        self.dump(pd.DataFrame([
            {"Score": score, "Elapsed": elapsed}
        ], index=[self.dataset]))


def main():
    # UCR datasets

    #MULT_SETS = [
    #    "ArticularyWordRecognition",
    #    "AtrialFibrillation", "BasicMotions", "CharacterTrajectories",
    #    "Cricket", "DuckDuckGeese", "EigenWorms", "Epilepsy",
    #    "ERing", "EthanolConcentration", "FaceDetection", "FingerMovements",
    #    "HandMovementDirection", "Handwriting", "Heartbeat",
    #    "InsectWingbeat", "JapaneseVowels", "Libras", "LSST", "MotorImagery",
    #    "NATOPS", "PEMS-SF", "PenDigits", "Phoneme", "Plots", "RacketSports",
    #    "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
    #    "StandWalkJump", "UWaveGestureLibrary",
    #]

    # small UEA datasets
    MULT_SETS = [
        "Libras",
        "AtrialFibrillation",
        "ERing",
        "BasicMotions",
        "RacketSports",
        "PenDigits",
        "Epilepsy",
        "JapaneseVowels",
        "StandWalkJump",
        "FingerMovements",
        "UWaveGestureLibrary",
        "Handwriting",
        "NATOPS",
    ]
#10216	HandMovementDirection
#11484	Plots
#15200	ArticularyWordRecognition
#20452	CharacterTrajectories
#26256	LSST
#34748	SelfRegulationSCP2
#34996	SelfRegulationSCP1
#42212	Cricket
#139304	SpokenArabicDigits
#264688	Phoneme
#303380	Heartbeat
#622688	DuckDuckGeese
#772016	FaceDetection
#827440	EigenWorms
#838112	PEMS-SF
#848340	EthanolConcentration
#1045712	MotorImagery
#1123400	InsectWingbeat

    luigi.build(
        [RunVotingEnsemble(dataset=dataset, levels=[2,3,4,5,6,7,8,9,10]) for dataset in DATASETS],
        workers=1,
        local_scheduler=True
    )
    #luigi.build(
    #    [RunLogit(dataset=dataset) for dataset in DATASETS],
    #    workers=4,
    #    local_scheduler=True
    #)
    # don't run multivariate for now, large memory usage
    #luigi.build(
    #    [RunMultivariate(dataset=d, levels=[2,3,4]) for d in MULT_SETS],
    #    workers=1, local_scheduler=True
    #)

if __name__ == "__main__":
    main()
