import evaluate
import pdb

class Evaluator:
    def __init__(self, references, predictions):
        self.references = references
        self.predictions = predictions
        self.results = {}

    def evaluate(self):
        self.add_rouge_score_to_result()
        self.add_bert_score_to_result()

    def add_rouge_score_to_result(self):
        rouge1, rouge2, rougeL = self.calculate_rouge_score()
        self.results["rouge1"] = rouge1
        self.results["rouge2"] = rouge2
        self.results["rougeL"] = rougeL

    def calculate_rouge_score(self) -> list[float]:
        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=self.predictions, references=self.references)
        return [scores["rouge1"], scores["rouge2"], scores["rougeL"]]
    
    def add_bert_score_to_result(self) -> float:
        bertscore = self.calculate_bert_score()
        self.results["bertscore"] = bertscore

    def calculate_bert_score(self) -> float:
        bertscore = evaluate.load("bertscore")
        scores = bertscore.compute(predictions=self.predictions, references=self.references, lang="en")
        return sum(scores["f1"]) / len(scores["f1"])


gold_standard = ["This is a summary of an article", "Testing 1 2 3"]
predictions = ["This is a somewhat longer summary of some article", "Testing 3 2 1"]
evaluator = Evaluator(references=gold_standard, predictions=predictions)
evaluator.evaluate()
print(evaluator.results)