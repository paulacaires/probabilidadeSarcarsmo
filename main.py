import pandas as pd
import sacrebleu
from bert_score import score as bertscore
import textstat
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
from nltk.translate.gleu_score import sentence_gleu

DATA_PATH = "dados/textos.csv"
RESULT_PATH = "dados/textos_resultado.csv"

df = pd.read_csv(DATA_PATH, sep=",", quotechar='"', engine="python", on_bad_lines="skip")
df.columns = ["original", "simplified"]

print("Loading models...")
nlp = spacy.load("pt_core_news_sm")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_bleu(orig, simp):
    """Compute BLEU using sacreBLEU."""
    # return sacrebleu.sentence_bleu(simp, [orig]).score / 100

def compute_bert_score(orig_list, simp_list):
    """Compute BERTScore for all pairs (vectorized)."""
    P, R, F1 = bertscore(
        simp_list, 
        orig_list, 
        lang="pt",
        verbose=True
    )
    return F1.tolist()

def compute_fkgl(text):
    """Compute Flesch-Kincaid Grade Level (readability)."""
    return textstat.flesch_kincaid_grade(text)

def count_sentences(text):
    """Count sentences using spaCy."""
    return len(list(nlp(text).sents))

def compute_cosine_similarity(orig, simp):
    """Compute semantic similarity using Sentence-BERT embeddings."""
    emb1 = embedder.encode(orig, convert_to_tensor=True)
    emb2 = embedder.encode(simp, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).cpu().numpy())

def compute_gleu(orig, simp):
    """Compute sentence-level GLEU score."""
    ref_tokens = orig.split()
    hyp_tokens = simp.split()
    try:
        score = sentence_gleu([ref_tokens], hyp_tokens)
    except ZeroDivisionError:
        score = 0.0
    return score

print("Computing metrics...")

# BLEU
df["BLEU"] = [compute_bleu(o, s) for o, s in tqdm(zip(df["original"], df["simplified"]), total=len(df), desc="BLEU")]

# BERTScore
df["BERTScore_F1"] = compute_bert_score(df["original"].tolist(), df["simplified"].tolist())

# GLEU
df["GLEU"] = [compute_gleu(o, s) for o, s in tqdm(zip(df["original"], df["simplified"]), total=len(df), desc="GLEU")]

# FKGL
df["FKGL_original"] = df["original"].apply(compute_fkgl)
df["FKGL_simplified"] = df["simplified"].apply(compute_fkgl)
df["FKGL_diff"] = df["FKGL_original"] - df["FKGL_simplified"]

# Structural change (SAMSA proxy)
df["num_sent_original"] = df["original"].apply(count_sentences)
df["num_sent_simplified"] = df["simplified"].apply(count_sentences)
df["structural_change"] = df["num_sent_simplified"] - df["num_sent_original"]

numeric_cols = [
    "BLEU", "GLEU", "BERTScore_F1", "FKGL_original", "FKGL_simplified",
    "FKGL_diff", "structural_change"
]

df[numeric_cols] = df[numeric_cols].applymap(lambda x: round(x, 4) if isinstance(x, (float, int)) else x)

os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
df.to_csv(RESULT_PATH, index=False)
print(f"\n✅ Metrics saved to: {RESULT_PATH}")

print("\n=== MÉDIAS E DESVIOS PADRÃO ===")

# DataFrame resumo com média e desvio padrão
summary = pd.DataFrame({
    "Média": df[numeric_cols].mean(),
    "Desvio Padrão": df[numeric_cols].std(),
    "Mediana": df[numeric_cols].median(),
    "Mínimo": df[numeric_cols].min(),
    "Máximo": df[numeric_cols].max()
}).round(4)

# Reorganiza colunas (só para exibição mais bonita)
summary = summary[["Média", "Desvio Padrão", "Mediana", "Mínimo", "Máximo"]]

print(summary)

# Salva resumo
os.makedirs("dados", exist_ok=True)
summary_path = os.path.join("dados", "metricas_resumo.csv")
summary.to_csv(summary_path, index=True)
print(f"\n📊 Resumo salvo em: {summary_path}")
