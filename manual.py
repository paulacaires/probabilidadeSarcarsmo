import pandas as pd

# Caminhos de entrada e saída
avaliacoes_path = "dados/avaliacoes_manuais.csv"
output_path_frases = "dados/avaliacoes_agrupadas.csv"
output_path_geral = "dados/avaliacoes_geral.csv"

# === Carregar dados ===
avaliacoes = pd.read_csv(avaliacoes_path)

# === Análises básicas ===
num_frases = avaliacoes["id_frase"].nunique()
print(f"Total de frases avaliadas: {num_frases}")

avaliacoes_por_frase = avaliacoes["id_frase"].value_counts().sort_index()
print("\nAvaliações por frase:")
print(avaliacoes_por_frase)

media_avaliacoes = avaliacoes_por_frase.mean()
desvio_avaliacoes = avaliacoes_por_frase.std()
print(f"\nMédia de avaliações por frase: {media_avaliacoes:.2f} ± {desvio_avaliacoes:.2f}")

# === MÉDIAS E DESVIOS POR FRASE ===
medias = (
    avaliacoes.groupby("id_frase")[["clareza", "fidelidade", "efetividade"]]
    .agg(["mean", "std"])
    .reset_index()
)

# Ajustar nomes das colunas
medias.columns = [
    "id_frase",
    "media_clareza", "std_clareza",
    "media_fidelidade", "std_fidelidade",
    "media_efetividade", "std_efetividade"
]

# Arredondar para 2 casas decimais
medias = medias.round(2)

# Salvar resultado por frase
medias.to_csv(output_path_frases, index=False)
print(f"\n✅ Arquivo salvo com médias por frase em: {output_path_frases}")

# === TABELA GERAL (TODAS AS AVALIAÇÕES) ===
geral = pd.DataFrame({
    "media_clareza": [avaliacoes["clareza"].mean()],
    "std_clareza": [avaliacoes["clareza"].std()],
    "media_fidelidade": [avaliacoes["fidelidade"].mean()],
    "std_fidelidade": [avaliacoes["fidelidade"].std()],
    "media_efetividade": [avaliacoes["efetividade"].mean()],
    "std_efetividade": [avaliacoes["efetividade"].std()]
}).round(2)

# Salvar tabela geral
geral.to_csv(output_path_geral, index=False)
print(f"✅ Arquivo salvo com médias gerais em: {output_path_geral}")

# Exibir prévias
print("\nPrévia das médias por frase:")
print(medias.head())

print("\nResumo geral (todas as frases):")
print(geral)
