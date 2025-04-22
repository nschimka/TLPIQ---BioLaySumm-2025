import evaluate
import pdb
from readability import Readability

class Evaluator:
    def __init__(self, references, predictions):
        self.references = references
        self.predictions = predictions
        self.results = {}

    def evaluate(self):
        # relevance metrics
        print("calculating rouge...")
        self.add_rouge_score_to_result()

        print("calculating bertscore...")
        self.add_bert_score_to_result()

        print("calculating meteor...")
        self.add_meteor_score_to_result()

        print("calculating bleu...")
        self.add_bleu_score_to_result()

        # readability metrics
        print("calculating Flesch-Kincaid, Dale-Chall, Coleman-Liau Index readability...")
        self.add_most_readability_scores_to_result()
        # TODO: LENS https://github.com/Yao-Dou/LENS

        # factuality metrics
        # TODO: alignscore https://github.com/yuh-zha/AlignScore
        # TODO: Summac https://pypi.org/project/summac/0.0.1/

    def add_rouge_score_to_result(self):
        rouge1, rouge2, rougeL = self.calculate_rouge_score()
        self.results["rouge1"] = rouge1
        self.results["rouge2"] = rouge2
        self.results["rougeL"] = rougeL

    """
    Evaluate quality of summary by comparing ngrams in reference text and output text. Emphasises recall over precision
    ROUGE-L calculates the longest common subsequence (LCS) between the reference and output summaries and is slightly more flexible
    """
    def calculate_rouge_score(self) -> list[float]:
        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=self.predictions, references=self.references)
        return [scores["rouge1"], scores["rouge2"], scores["rougeL"]]
    
    def add_bert_score_to_result(self):
        bertscore = self.calculate_bert_score()
        self.results["bertscore"] = bertscore

    """
    BERTScore takes the pre-trained contextual embeddings built by BERT and uses them to match words in candidate and reference sentences by cosine similarity.
    In other words, it calculates the similarity between the two. It correlates with human judgment on sentence-level and system-level evaluation.
    """
    def calculate_bert_score(self) -> float:
        bertscore = evaluate.load("bertscore")
        scores = bertscore.compute(predictions=self.predictions, references=self.references, lang="en")
        return sum(scores["f1"]) / len(scores["f1"])
    
    def add_meteor_score_to_result(self):
        meteor = self.calculate_meteor_score()
        self.results["meteor"] = meteor
    
    """
    Matches generalized unigrams (matched on surface or stem forms and meanings) between reference and candidate translations.
    Correlates well with human judgments and balances recall and precision.
    """
    def calculate_meteor_score(self) -> float:
        meteor = evaluate.load("meteor")
        return meteor.compute(predictions=self.predictions, references=self.references)["meteor"]
    
    def add_bleu_score_to_result(self):
        bleu = self.calculate_bleu_score()
        self.results["bleu"] = bleu

    """
    'the closer a machine translation is to a professional human translation, the better it is' - commonly used for machine translation
    average per-sentence scores of shared n-grams over the whole corpus. Emphasizes precision over recall
    """
    def calculate_bleu_score(self) -> float:
        bleu = evaluate.load("bleu")
        return bleu.compute(predictions=self.predictions, references=self.references)["bleu"]
    
    def add_most_readability_scores_to_result(self):
        fkgl, dcrs, cli = self.calculate_most_readability_scores()
        self.results["flesh_kincaid"] = fkgl
        self.results["dale_chall"] = dcrs
        self.results["coleman_liau"] = cli

    def calculate_most_readability_scores(self) -> list[float]:
        total_prediction = " ".join(self.predictions)
        r = Readability(total_prediction)
        return [r.flesch_kincaid().score, r.dale_chall().score, r.coleman_liau().score]
        


gold_standard = ["Kidney function depends on the nephron , which comprises a blood filter , a tubule that is subdivided into functionally distinct segments , and a collecting duct . How these regions arise during development is poorly understood . The zebrafish pronephros consists of two linear nephrons that develop from the intermediate mesoderm along the length of the trunk . Here we show that , contrary to current dogma , these nephrons possess multiple proximal and distal tubule domains that resemble the organization of the mammalian nephron . We examined whether pronephric segmentation is mediated by retinoic acid ( RA ) and the caudal ( cdx ) transcription factors , which are known regulators of segmental identity during development . Inhibition of RA signaling resulted in a loss of the proximal segments and an expansion of the distal segments , while exogenous RA treatment induced proximal segment fates at the expense of distal fates . Loss of cdx function caused abrogation of distal segments , a posterior shift in the position of the pronephros , and alterations in the expression boundaries of raldh2 and cyp26a1 , which encode enzymes that synthesize and degrade RA , respectively."]
predictions = ["In the kidney , structures known as nephrons are responsible for collecting metabolic waste . Nephrons are composed of a blood filter ( glomerulus ) followed by a series of specialized tubule regions , or segments , which recover solutes such as salts , and finally terminate with a collecting duct . The genetic mechanisms that establish nephron segmentation in mammals have been a challenge to study because of the kidney's complex organogenesis . The zebrafish embryonic kidney ( pronephros ) contains two nephrons , previously thought to consist of a glomerulus , short tubule , and long stretch of duct . In this study , we have redefined the anatomy of the zebrafish pronephros and shown that the duct is actually subdivided into distinct tubule segments that are analogous to the proximal and distal segments found in mammalian nephrons . Next , we used the zebrafish pronephros to investigate how nephron segmentation occurs . We found that retinoic acid ( RA ) induces proximal pronephros segments and represses distal segment fates . Further , we found that the caudal ( cdx ) transcription factors direct the anteroposterior location of pronephric progenitors by regulating the site of RA production . Taken together , these results reveal that a cdx-RA pathway plays a key role in both establishing where the pronephros forms along the embryonic axis as well as its segmentation pattern ."]
evaluator = Evaluator(references=gold_standard, predictions=predictions)
evaluator.evaluate()
print(evaluator.results)