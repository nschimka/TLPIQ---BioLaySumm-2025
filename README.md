https://biolaysumm.org/
Preprocessing
The preprocessing pipeline prepares scientific articles from the PLOS and eLife datasets for the biomedical lay summarization task.
0.preprocess.py
This script extracts key sections from scientific articles (abstract, introduction, results, discussion, methods), removes citations, simplifies technical terminology, and formats the text with appropriate section markers. It handles token limitations by intelligently truncating less important sections while preserving complete sentences.
Usage:
bashpython src/0.preprocess.py --output_dir /data --dataset both --max_tokens 1024 --batch_size 32 --output_format jsonl
Key parameters:

--output_dir: Directory to save processed data
--dataset: Which dataset to process (plos, elife, or both)
--max_tokens: Maximum token limit per article (default: 1024)
--batch_size: Batch size for processing (adjust based on available memory)
--output_format: Output format (jsonl or csv)

The script produces structured files containing preprocessed articles with their corresponding summaries, ready for model training and evaluation.
Evaluation
There are some conflicting dependencies in the evaluation systems; for example, LENS and AlignScore want different versions of pytorch-lightning. You'll get some error messages when installing but they don't seem to break anything. Follow these steps to get everything set up:

Make sure your system has the wget command available for the terminal; if you're on a Mac with Homebrew, you can brew install wget
pip install -r requirements-eval.txt to install all dependencies.
Run bash get_models.sh to clone and install AlignScore

Note that while all the metrics can run on a CPU, the model-based ones (AlignScore, Summac, especially LENS) will want CUDA-enabled GPU to run efficiently (https://pytorch.org/get-started/locally/).
To test that everything is working, you can run the evaluations in test mode: python src/3.eval.py --test_mode. If that works, you can run with the actual data using python src/3.eval.py. It assumes that you've put CSV prediction files within data with the following columns: input_text (the article or a variation of it), summary (the gold standard reference), and generated_summary (the model output).
The script runs separately on the eLife and PLOS datasets. Realistically you will want to comment them out and run on one dataset at a time, otherwise you'll lose the printed results in your terminal.