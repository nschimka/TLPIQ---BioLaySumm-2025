# BioLaySumm 2025

https://biolaysumm.org/

## Evaluation

There are some conflicting dependencies in the evaluation systems; for example, LENS and AlignScore want different versions of pytorch-lightning. I recommend using `pip install -r requirements-eval.txt` to install all the dependencies but be aware you'll probably get some errors in the terminal about invalid dependencies. Just try `pip install X` if the package seems to be missing and run the evaluation script.

Note that LENS only seems to support CUDA devices (https://pytorch.org/get-started/locally/) because the library doesn't provide a way to pass "cpu" as a device when loading the model (at least that I can see). Some of the metrics will probably run faster on a GPU anyway. Your device also needs to have the `wget` command to download files; if you're on a Mac with Homebrew, you can `brew install wget`.

To run evaluations, first run `bash 573/src/visualization/get_models.sh` to clone and install AlignScore, then run `python 573/src/visualization/evaluation.py --test_mode`. If that works, you can run with the actual data using `python 573/src/visualization/evaluation.py`. It assumes that you've downloaded the datasets already and put them in `573/data/external`.
