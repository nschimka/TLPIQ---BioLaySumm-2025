# BioLaySumm 2025

https://biolaysumm.org/

## Evaluation

There are some conflicting dependencies in the evaluation systems; for example, LENS and AlignScore want different versions of pytorch-lightning. You'll get some error messages when installing but they don't seem to break anything. Follow these steps to get everything set up:

1) Make sure your system has the `wget` command available for the terminal; if you're on a Mac with Homebrew, you can `brew install wget`
2) `pip install -r requirements-eval.txt` to install all dependencies.
3) Run `bash get_models.sh` to clone and install AlignScore

Note that while all the metrics can run on a CPU, the model-based ones (AlignScore, Summac, especially LENS) will want CUDA-enabled GPU to run efficiently (https://pytorch.org/get-started/locally/).

To test that everything is working, you can run the evaluations in test mode: `python src/3.eval.py --test_mode`. If that works, you can run with the actual data using `python src/3.eval.py`. It assumes that you've put CSV prediction files within `data` with the following columns: `input_text` (the article or a variation of it), `summary` (the gold standard reference), and `generated_summary` (the model output).

The script runs separately on the eLife and PLOS datasets. Realistically you will want to comment them out and run on one dataset at a time, otherwise you'll lose the printed results in your terminal.
