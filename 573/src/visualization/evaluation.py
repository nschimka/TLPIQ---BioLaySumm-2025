import evaluate
import numpy as np
import pdb
from readability import Readability
from lens import download_model, LENS
import torch
from summac.model_summac import SummaCConv
import nltk

# reference: https://github.com/TGoldsack1/BioLaySumm2024-evaluation_scripts/blob/master/evaluate.py
class Evaluator:
    def __init__(self, references, predictions, articles):
        self.references = references
        self.predictions = predictions
        self.articles = articles
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.results = {}

    def evaluate(self):
        # relevance metrics
        print("calculating rouge...")
        #self.add_rouge_score_to_result()

        print("calculating bertscore...")
        #self.add_bert_score_to_result()

        print("calculating meteor...")
        #self.add_meteor_score_to_result()

        print("calculating bleu...")
        #self.add_bleu_score_to_result()

        # readability metrics
        print("calculating Flesch-Kincaid, Dale-Chall, Coleman-Liau Index readability...")
        #self.add_most_readability_scores_to_result()

        print("calculating LENS...")
        # LENS https://github.com/Yao-Dou/LENS
        #self.add_lens_score_to_result()

        # factuality metrics
        # TODO: alignscore https://github.com/yuh-zha/AlignScore
        self.add_summac_score_to_result()

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
        return np.mean(scores["f1"])
    
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
        fkcg_scores = []
        dcrs_scores = []
        cli_scores = []
        for prediction in self.predictions:
            readability = Readability(prediction)
            fkcg_scores.append(readability.flesch_kincaid().score)
            dcrs_scores.append(readability.dale_chall().score)
            cli_scores.append(readability.coleman_liau().score)
        return [np.mean(fkcg_scores), np.mean(dcrs_scores), np.mean(cli_scores)]
    
    def add_lens_score_to_result(self):
        lens = self.calculate_lens_score()
    
    def calculate_lens_score(self) -> float:
        # lot of GPUs aren't CUDA compatible:  https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with/61034368#61034368
        DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #DEVICES = [0] if torch.cuda.is_available() else None
        
        lens_path = download_model("davidheineman/lens")

        # Original LENS is a real-valued number. 
        # Rescaled version (rescale=True) rescales LENS between 0 and 100 for better interpretability. 
        # You can also use the original version using rescale=False

        lens = LENS(lens_path, rescale=True)

        complex = self.articles
        simple = self.predictions
        references = self.references

        breakpoint()

        return np.mean(lens.score(complex, simple, references, batch_size=8, devices=DEVICES))
    
    def add_summac_score_to_result(self):
        summac = self.calculate_summac_score()
        self.result["summac"] = summac
     
    # wget needs to be installed. On mac: brew install wget
    def calculate_summac_score(self):
        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=self.device, start_file="default", agg="mean")
        return np.mean(model_conv.score(self.references, self.predictions)['scores'])
                
# if this fails with an SSL error on Mac, bash '/Applications/Python 3.10/Install Certificates.command'
nltk.download("punkt_tab")

articles = ["Kidney function depends on the nephron , which comprises a blood filter , a tubule that is subdivided into functionally distinct segments , and a collecting duct . How these regions arise during development is poorly understood . The zebrafish pronephros consists of two linear nephrons that develop from the intermediate mesoderm along the length of the trunk . Here we show that , contrary to current dogma , these nephrons possess multiple proximal and distal tubule domains that resemble the organization of the mammalian nephron . We examined whether pronephric segmentation is mediated by retinoic acid ( RA ) and the caudal ( cdx ) transcription factors , which are known regulators of segmental identity during development . Inhibition of RA signaling resulted in a loss of the proximal segments and an expansion of the distal segments , while exogenous RA treatment induced proximal segment fates at the expense of distal fates . Loss of cdx function caused abrogation of distal segments , a posterior shift in the position of the pronephros , and alterations in the expression boundaries of raldh2 and cyp26a1 , which encode enzymes that synthesize and degrade RA , respectively.", "White-nose syndrome is one of the most lethal wildlife diseases , killing over 5 million North American bats since it was first reported in 2006 . The causal agent of the disease is a psychrophilic filamentous fungus , Pseudogymnoascus destructans . The fungus is widely distributed in North America and Europe and has recently been found in some parts of Asia , but interestingly , no mass mortality is observed in European or Asian bats . Here we report a novel double-stranded RNA virus found in North American isolates of the fungus and show that the virus can be used as a tool to study the epidemiology of White-nose syndrome . The virus , termed Pseudogymnoascus destructans partitivirus-pa , contains 2 genomic segments , dsRNA 1 and dsRNA 2 of 1 . 76 kbp and 1 . 59 kbp respectively , each possessing a single open reading frame , and forms isometric particles approximately 30 nm in diameter , characteristic of the genus Gammapartitivirus in the family Partitiviridae . Phylogenetic analysis revealed that the virus is closely related to Penicillium stoloniferum virus S . We were able to cure P ."]
gold_standards = ["In the kidney , structures known as nephrons are responsible for collecting metabolic waste . Nephrons are composed of a blood filter ( glomerulus ) followed by a series of specialized tubule regions , or segments , which recover solutes such as salts , and finally terminate with a collecting duct . The genetic mechanisms that establish nephron segmentation in mammals have been a challenge to study because of the kidney's complex organogenesis . The zebrafish embryonic kidney ( pronephros ) contains two nephrons , previously thought to consist of a glomerulus , short tubule , and long stretch of duct . In this study , we have redefined the anatomy of the zebrafish pronephros and shown that the duct is actually subdivided into distinct tubule segments that are analogous to the proximal and distal segments found in mammalian nephrons . Next , we used the zebrafish pronephros to investigate how nephron segmentation occurs . We found that retinoic acid ( RA ) induces proximal pronephros segments and represses distal segment fates . Further , we found that the caudal ( cdx ) transcription factors direct the anteroposterior location of pronephric progenitors by regulating the site of RA production . Taken together , these results reveal that a cdx-RA pathway plays a key role in both establishing where the pronephros forms along the embryonic axis as well as its segmentation pattern .", "Many species of bats in North America have been severely impacted by a fungal disease , white-nose syndrome , that has killed over 5 million bats since it was first identified in 2006 . The fungus is believed to have been introduced into a cave in New York where bats hibernate , and has now spread to 29 states and 4 Canadian provinces . The fungus is nearly identical from all sites where it has been isolated; however , we discovered that the fungus harbors a virus , and the virus varies enough to be able to use it to understand how the fungus has been spreading . This study used samples from infected bats throughout Pennsylvania and New York , and New Brunswick , Canada , as well a few isolates from other northeastern states . The evolution of the virus recapitulates the spread of the virus across these geographical areas , and should be useful for studying the further spread of the fungus ."]
predictions = ["In the kidney , tiny units called nephrons remove waste from the blood . Each nephron has a filter ( called the glomerulus ) , followed by different tube sections that reabsorb useful substances like salts , and ends with a collecting duct . Studying how these sections form in mammals is difficult because kidney development is very complex . In zebrafish embryos , their simple kidneys ( called pronephros ) were once thought to just have a glomerulus , a short tube , and a long duct . But in this study , researchers found that this ' duct ' is actually made up of several distinct parts , similar to the sections seen in mammal kidneys . The researchers then looked into how these parts form . They discovered that a substance called retinoic acid ( RA ) helps form the front ( proximal ) parts of the kidney and blocks the formation of the back ( distal ) parts . They also found that certain genes called cdx control where the kidney forms along the body by influencing where RA is made . In summary , they showed that the cdx genes and RA work together to decide both where the kidney forms in the embryo and how its parts are organized .", "Many types of bats in North America have been badly affected by a disease called white-nose syndrome , caused by a fungus . Since it was first found in 2006 , the disease has killed over 5 million bats . It likely started in a New York cave where bats hibernate and has now spread to 29 U . S . states and 4 provinces in Canada . Even though the fungus looks almost the same everywhere it ' s found , researchers discovered that it carries a virus , and this virus changes slightly depending on the location . These small changes help scientists track how the fungus is spreading . In this study , scientists used samples from bats in Pennsylvania , New York , New Brunswick ( Canada ) , and a few other northeastern states . They found that the virus ’ s changes match the way the disease has spread across these areas . This means the virus can be a helpful tool for studying how the fungus continues to spread ."]
evaluator = Evaluator(references=gold_standards, predictions=predictions, articles=articles)
evaluator.evaluate()
print(evaluator.results)