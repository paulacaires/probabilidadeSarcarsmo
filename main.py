import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
import os
from sklearn.preprocessing import normalize

# Caminhos
MODELO_SENTENCE_PATH = "modelo/modelo_finetunado_sarcasmo"
CLASSIFICADOR_PATH = os.path.join(MODELO_SENTENCE_PATH, "classificador_logreg.pkl")
CSV_PATH = "dados/textos.csv"
CSV_SAIDA = "dados/resultados_sarcasmo.csv"


def carregar_modelos():
    print("🔹 Carregando modelo de embeddings...")
    model = SentenceTransformer(MODELO_SENTENCE_PATH)

    print("🔹 Carregando classificador...")
    clf = joblib.load(CLASSIFICADOR_PATH)
    return model, clf


def calcular_prob_sarcasmo(model, clf, textos):    
    # Cria embeddings em lote
    embedding = model.encode(textos, convert_to_tensor=True).cpu().tolist()
    # Calcula as probabilidades para todos os textos
    probas = clf.predict_proba(embedding)[:, 1]
    return probas    

def main():
    if not os.path.exists(MODELO_SENTENCE_PATH):
        raise FileNotFoundError("❌ Pasta do modelo SentenceTransformer não encontrada.")
    if not os.path.exists(CLASSIFICADOR_PATH):
        raise FileNotFoundError("❌ Arquivo do classificador não encontrado.")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("❌ Arquivo CSV de entrada não encontrado.")

    model, clf = carregar_modelos()

    print("Lendo planilha...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    col_antes = "frase_original"
    col_depois = "frase_reescrita"

    # --- Calcular probabilidades --- #
    print("🔹 Calculando probabilidade de sarcasmo (antes)...")
    df["prob_sarcasmo_antes"] = calcular_prob_sarcasmo(model, clf, df[col_antes].astype(str).tolist())

    print("🔹 Calculando probabilidade de sarcasmo (depois)...")
    df["prob_sarcasmo_depois"] = calcular_prob_sarcasmo(model, clf, df[col_depois].astype(str).tolist())

    # --- Salvar resultados --- #
    df.to_csv(CSV_SAIDA, index=False)
    print(f"✅ Resultados salvos em: {CSV_SAIDA}")


if __name__ == "__main__":
    main()
