from .classification import Waveform, Twonorm, Ringnorm, Musk2, Vowel, \
    BinaryVowel, Madelon, Hastie, Covertype, BinaryCovertype, Letters, \
    BinaryLetters, MNIST, MNIST8vs9, BinaryMNIST

from .regression import CTSlice, Friedman1, Cadata, Abalone, OzoneLA, \
    Diabetes, Hardware, BostonHousing, MPG

__DATASET__ = {
    # From Begon, J. M., Joly, A., & Geurts, P. (2017, July). Globally induced forest: A prepruning compression scheme. In International Conference on Machine Learning (pp. 420-428). PMLR.
    "waveform": Waveform,
    "twonorm": Twonorm,
    "ringnorm": Ringnorm,
    "musk2": Musk2,
    "vowel": Vowel,
    "binary_vowel": BinaryVowel,
    "madelon": Madelon,
    "hastie": Hastie,
    "covertype": Covertype,
    "binary_covertype": BinaryCovertype,
    "letters": Letters,
    "binary_letters": BinaryLetters,
    "mnist": MNIST,
    "mnist8vs9": MNIST8vs9,
    "binary_mnist": BinaryMNIST,
    "ct_slice": CTSlice,
    "friedman1": Friedman1,
    "cadata": Cadata,
    "abalone": Abalone,
    # To compare with Bénard, C., Biau, G., Veiga, S., & Scornet, E. (2021, March). Interpretable random forests via rule extraction. In International Conference on Artificial Intelligence and Statistics (pp. 937-945). PMLR.
    "ozone": OzoneLA,
    "diabetes": Diabetes,
    "hardware": Hardware,
    "housing": BostonHousing,
    "mpg": MPG
}

def is_regression(full_dataset):
    from .regression import RegressionFullDataset
    return isinstance(full_dataset, RegressionFullDataset)

def is_classification(full_dataset):
    from .classification import ClassificationFullDataset
    return isinstance(full_dataset, ClassificationFullDataset)

def is_binary_classification(full_dataset):
    from .classification import ClassificationFullDataset
    return isinstance(full_dataset, ClassificationFullDataset) and \
           full_dataset.n_classes == 2

def download_all(folder=None, verbose=True):
    for fullset_cls in __DATASET__.values():
        fullset = fullset_cls(folder)
        print("Downloading for {}".format(repr(fullset)))
        try:
            fullset.load()
            if verbose:
                X, y = fullset.training_set
                print("\t> Training set shapes: {}, {}".format(X.shape, y.shape))
                X, y = fullset.test_set
                print("\t> Test set shapes: {}, {}".format(X.shape, y.shape))
        except Exception as e:
            if not verbose:
                raise
            print("Error while downloading '{}' ({}). "
                  "Skipping...".format(fullset.__class__.__name__, e))


__all__ = ["__DATASET__", "download_all", "is_regression", "is_classification",
           "is_binary_classification"]
