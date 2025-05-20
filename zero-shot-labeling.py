import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Carregar o dataset
df = pd.read_parquet("data/dev_test.parquet")

# Definir o parser de saída JSON
output_parser = JsonOutputParser()

# Definir o prompt zero-shot com saída estruturada em JSON
prompt = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "Classifique a seguinte notícia em uma das categorias abaixo:\n\n"
        "- 'poder': Notícias relacionadas ao governo brasileiro, política nacional, "
        "Congresso Nacional, STF, eleições, partidos políticos, administração pública, "
        "leis e legislações brasileiras.\n"
        "- 'mercado': Notícias sobre economia, finanças, negócios, empresas, bolsa de valores, "
        "indicadores econômicos, inflação, desemprego ou setor privado.\n"
        "- 'esportes': Notícias sobre competições esportivas, atletas, times, "
        "campeonatos, jogos olímpicos, futebol, basquete ou qualquer modalidade esportiva.\n"
        "- 'mundo': Notícias internacionais, acontecimentos em outros países, "
        "relações diplomáticas, conflitos internacionais ou política externa.\n\n"
        "Notícia: {input_text}\n\n"
        "Responda APENAS com um objeto JSON no seguinte formato: {{\"categoria\": \"NOME_DA_CATEGORIA\"}}, "
        "onde NOME_DA_CATEGORIA deve ser uma das seguintes opções: poder, mercado, esportes ou mundo."
    ),
)

# Inicializar o modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Compor a cadeia de execução
chain = prompt | llm | output_parser

# Armazenar os rótulos
rotulos = []

total = len(df)
for i, texto in enumerate(df["text"], start=1):
    try:
        resposta = chain.invoke({"input_text": texto})
        # Extrair a categoria do JSON (resposta já é um dicionário)
        categoria = resposta.get("categoria", None).strip().lower()
        rotulos.append(categoria)
    except Exception as e:
        rotulos.append("erro")
        print(f"[{i}/{total}] Erro ao processar: {texto[:50]}... -> {e}")
        continue

    # Mostrar progresso a cada 10 registros
    if i % 10 == 0 or i == total:
        print(f"[{i}/{total}] Processado")

# Adicionar a nova coluna com os rótulos
df["label"] = rotulos

# Salvar o resultado
df.to_parquet("data/labeled/noticias_rotuladas_zeroshot.parquet", index=False)
print(df.head())