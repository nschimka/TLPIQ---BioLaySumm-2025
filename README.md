# BioLaySumm 2025

## Evaluation

Note that LENS only seems to support CUDA devices because the library doesn't provide a way to pass "cpu" as a device when loading the model. Your device also needs to have the `wget` command downloaded to download files; if you're on a Mac with Homebrew, you can `brew install wget`. To run evaluations, first run `bash get_models.sh` to clone and install AlignScore, then run `python evaluation.py`.

There are also some conflicting dependencies. For example, LENS and AlignScore want different versions of pytorch-lightning.

