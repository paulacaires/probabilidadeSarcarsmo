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
    "GLEU", "BERTScore_F1", "FKGL_original", "FKGL_simplified",
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

# === BOXPLOTS INDIVIDUAIS (um por métrica, com verificação robusta) ===
print("\n📈 Gerando boxplots individuais (um por métrica)...")

sns.set(style="whitegrid", palette="muted")

# Diretório de saída
output_dir = "dados/boxplots"
os.makedirs(output_dir, exist_ok=True)

for metric in numeric_cols:
    # Verifica se a coluna contém números válidos
    col_data = pd.to_numeric(df[metric], errors="coerce").dropna()
    
    if col_data.empty:
        print(f"⚠️  Métrica '{metric}' não contém dados numéricos válidos. Pulando...")
        continue
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=col_data, color="skyblue", fliersize=4, width=0.4)
    plt.title(f"Distribuição da métrica: {metric}", fontsize=12, fontweight="bold")
    plt.ylabel("Value", fontsize=11)
    plt.xlabel("")  # remove o eixo x (decorativo)
    plt.tight_layout()

    # Caminho para salvar o gráfico
    plot_path = os.path.join(output_dir, f"boxplot_{metric}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"✅ Boxplot salvo: {plot_path}")

print("\n📊 Todos os boxplots individuais foram gerados com sucesso!")

# === BOXPLOTS COM ESCALAS INDEPENDENTES (todos em uma imagem) ===
print("\n📊 Gerando painel único com todos os boxplots...")

sns.set(style="whitegrid", palette="muted")

# Ordem personalizada das métricas
ordered_metrics = [
    "GLEU",
    "BERTScore_F1",
    "structural_change",
    "FKGL_original",
    "FKGL_simplified",
    "FKGL_diff"
]

# Labels customizados
labels = {
    "GLEU": "GLEU",
    "BERTScore_F1": "BERTScore (F1)",
    "structural_change": "Structural Change",
    "FKGL_original": "FKLG (original)",
    "FKGL_simplified": "FKLG (rewritten)",
    "FKGL_diff": "FKLG (difference)"
}

# Grade fixa: 2 linhas × 3 colunas
rows = 2
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()

for i, metric in enumerate(ordered_metrics):
    col_data = pd.to_numeric(df[metric], errors="coerce").dropna()

    if col_data.empty:
        axes[i].text(0.5, 0.5, f"Sem dados\n({metric})", ha="center", va="center")
        axes[i].set_axis_off()
        continue

    sns.boxplot(y=col_data, ax=axes[i], color="skyblue", fliersize=4, width=0.4)
    axes[i].set_title(labels[metric], fontsize=12, fontweight="bold", pad=10)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Ajuste do espaçamento
plt.tight_layout(h_pad=2.5, w_pad=2.0)

panel_path = os.path.join("dados", "boxplots", "painel_boxplots_todas_metricas.png")
os.makedirs(os.path.dirname(panel_path), exist_ok=True)
plt.savefig(panel_path, dpi=300)
plt.show()

print(f"✅ Painel salvo em: {panel_path}")

