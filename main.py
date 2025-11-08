import pandas as pd
import sacrebleu
from bert_score import score as bertscore
import textstat
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from tqdm import tqdm
import os

DATA_PATH = "dados/textos.csv"
RESULT_PATH = "dados/textos_resultado.csv"

df = pd.read_csv(DATA_PATH, sep=",", quotechar='"', engine="python", on_bad_lines="skip")
df.columns = ["original", "simplified"]

print("Loading models...")
nlp = spacy.load("pt_core_news_sm")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_bleu(orig, simp):
    """Compute BLEU using sacreBLEU."""
    return sacrebleu.corpus_bleu([simp], [[orig]]).score

def compute_bert_score(orig_list, simp_list):
    """Compute BERTScore for all pairs (vectorized)."""
    P, R, F1 = bertscore(simp_list, orig_list, lang="pt", verbose=True)
    return F1.tolist()

def compute_fkgl(text):
    """Compute Flesch-Kincaid Grade Level (readability)."""
    return textstat.flesch_kincaid_grade(text)

def count_sentences(text):
    """Count sentences using spaCy."""
    return len(list(nlp(text).sents))

def compute_rougeL(orig, simp):
    """Compute ROUGE-L (Longest Common Subsequence)."""
    return rouge.score(orig, simp)["rougeL"].fmeasure

def compute_cosine_similarity(orig, simp):
    """Compute semantic similarity using Sentence-BERT embeddings."""
    emb1 = embedder.encode(orig, convert_to_tensor=True)
    emb2 = embedder.encode(simp, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).cpu().numpy())

print("Computing metrics...")

# BLEU
df["BLEU"] = [compute_bleu(o, s) for o, s in tqdm(zip(df["original"], df["simplified"]), total=len(df), desc="BLEU")]

# BERTScore
df["BERTScore_F1"] = compute_bert_score(df["original"].tolist(), df["simplified"].tolist())

# FKGL
df["FKGL_original"] = df["original"].apply(compute_fkgl)
df["FKGL_simplified"] = df["simplified"].apply(compute_fkgl)
df["FKGL_diff"] = df["FKGL_original"] - df["FKGL_simplified"]

# Structural change (SAMSA proxy)
df["num_sent_original"] = df["original"].apply(count_sentences)
df["num_sent_simplified"] = df["simplified"].apply(count_sentences)
df["structural_change"] = df["num_sent_simplified"] - df["num_sent_original"]

# ROUGE-L
df["ROUGE-L"] = [compute_rougeL(o, s) for o, s in tqdm(zip(df["original"], df["simplified"]), total=len(df), desc="ROUGE-L")]

# Cosine similarity
df["Cosine_Similarity"] = [compute_cosine_similarity(o, s) for o, s in tqdm(zip(df["original"], df["simplified"]), total=len(df), desc="Cosine")]

numeric_cols = ["BLEU", "BERTScore_F1", "FKGL_original", "FKGL_simplified",
                "FKGL_diff", "structural_change", "ROUGE-L", "Cosine_Similarity"]

df[numeric_cols] = df[numeric_cols].applymap(lambda x: round(x, 4) if isinstance(x, (float, int)) else x)

os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
df.to_csv(RESULT_PATH, index=False)
print(f"\n✅ Metrics saved to: {RESULT_PATH}")

print("\n=== AVERAGE SCORES ===")
for metric in numeric_cols:
    print(f"{metric}: {df[metric].mean():.4f}")

sns.set(style="whitegrid")

# Boxplot of distributions
plt.figure(figsize=(10,6))
df_melt = df[numeric_cols].melt(var_name="Metric", value_name="Score")
sns.boxplot(x="Metric", y="Score", data=df_melt)
plt.title("Distribution of Evaluation Metrics")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dados/metrics_boxplot.png")
plt.show()

# Correlation matrix
plt.figure(figsize=(8,6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Evaluation Metrics")
plt.tight_layout()
plt.savefig("dados/metrics_corr.png")
plt.show()
