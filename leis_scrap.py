import requests
from bs4 import BeautifulSoup
import os
import re
import json

# Inicialmente ia fazer um script mais automatizado em relação a isso, mas
# percebi que ia ser
# mais rápido e fácil só escolher e pegar manualmente o link um a um do que
# fazer requests+soup
#
# Em trabalhos futuros, considerar algum método mais robusto para rodar
# todas as leis e baixar todas. Por mais bare-bones que o site seja,
# fazem manutenção nele e ainda atualizam os links, então precisaria de um
# meio pra visitar todos os links possíveis partindo dele para criar uma base
# realmente completa

# NOTE Para trabalhos futuros, alguns urls com a lei compilada são sufixado
# por 'compilada',
# NOTE enquanto outros por 'compilado'. Não parece haver um padrão pra isso
# NOTE https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/lei/L13105compilado.htm 
# aparentemente movido durante o desenvolvimento

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

LEIS = [
    {
        "nome_popular": "Constituição Federal de 1988",
        "numero": "CF/1988",
        "ano": 1988,
        "url": "https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm"  # funcional
    },
    {
        "nome_popular": "Código Civil",
        "numero": "Lei 10.406/2002",
        "ano": 2002,
        "url": "https://www.planalto.gov.br/ccivil_03/leis/2002/L10406compilada.htm"  # funcional
    },
    {
        "nome_popular": "Código Penal",
        "numero": "Decreto-Lei 2.848/1940",
        "ano": 1940,
        "url": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del2848compilado.htm"  # funcional
    },
    {
        "nome_popular": "Código de Processo Penal",
        "numero": "Decreto-Lei 3.689/1941",
        "ano": 1941,
        "url": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del3689compilado.htm"  # funcional
    },
    {
        "nome_popular": "Código de Processo Civil",
        "numero": "Lei 13.105/2015",
        "ano": 2015,
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/lei/L13105compilado.htm"  # funcional
    },
    {
        "nome_popular": "Código Tributário Nacional",
        "numero": "Lei 5.172/1966",
        "ano": 1966,
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l5172compilado.htm"  # funcional
    },
    {
        "nome_popular": "Código de Defesa do Consumidor",
        "numero": "Lei 8.078/1990",
        "ano": 1990,
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm"  # funcional
    },
    {
        "nome_popular": "Consolidação das Leis do Trabalho – CLT",
        "numero": "Decreto-Lei 5.452/1943",
        "ano": 1943,
        "url": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452compilado.htm" # funcional
    },
    {
        "nome_popular": "Estatuto da Criança e do Adolescente (ECA)",
        "numero": "Lei 8.069/1990",
        "ano": 1990,
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l8069compilado.htm"  # funcional
    },
    {
        "nome_popular": "Estatuto do Idoso",
        "numero": "Lei 10.741/2003",
        "ano": 2003,
        "url": "https://www.planalto.gov.br/ccivil_03/leis/2003/L10.741compilado.htm"  # funcional
    },
    {
        "nome_popular": "Estatuto da Igualdade Racial",
        "numero": "Lei 12.288/2010",
        "ano": 2010,
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2007-2010/2010/lei/L12288.htm"  # funcional
    },
    {
        "nome_popular": "Lei Maria da Penha",
        "numero": "Lei 11.340/2006",
        "ano": 2006,
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2006/lei/l11340.htm"  # funcional
    },
    {
        "nome_popular": "Lei de Drogas",
        "numero": "Lei 11.343/2006",
        "ano": 2006,
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2006/lei/l11343.htm#view" # funcional
    },
    {
        "nome_popular": "Lei de Acesso à Informação",
        "numero": "Lei 12.527/2011",
        "ano": 2011,
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12527.htm" # funcional
    },
    {
        "nome_popular": "Lei Geral de Proteção de Dados (LGPD)",
        "numero": "Lei 13.709/2018",
        "ano": 2018,
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709compilado.htm" # funcional
    }
]

PASTA_JSON = "leis_json"
PASTA_TXT = "leis_txt"
os.makedirs(PASTA_JSON, exist_ok=True)
os.makedirs(PASTA_TXT, exist_ok=True)


def limpar_texto(html_element):
    """
    Extrai e normaliza o texto de um elemento HTML:
    - Remove múltiplos espaços
    - Concatena '1 o' ou '1 <sup><u>o</u></sup>' -> '1º'
    - Normaliza quebras de linha
    """
    # Pega o texto bruto, mantendo espaçamento
    texto = html_element.get_text(" ", strip=True)

    # Remove múltiplos espaços
    texto = re.sub(r"\s+", " ", texto)

    # Normaliza número + "o" solto (1 o -> 1º, 2 o -> 2º)
    texto = re.sub(r"(\d+)\s*o\b", r"\1º", texto)

    # Normaliza <sup><u>o</u></sup> ou <sup>o</sup>
    texto = re.sub(r"(\d+)\s*[ºo]", r"\1º", texto)

    # Caso apareça "Art. 1o" -> "Art. 1º"
    texto = re.sub(r"(Art\.\s*\d+)o\b", r"\1º", texto, flags=re.IGNORECASE)

    return texto.strip()


def extrair_estrutura(soup, lei_nome, lei_numero, url):
    blocos = soup.find_all(["p"])  # pode expandir com div/span se necessário

    unidades = []
    artigo_atual = None

    for bloco in blocos:
        linha = limpar_texto(bloco)
        if not linha:
            continue

        # ---------------------
        # Detecta Artigos
        # ---------------------
        artigo_match = re.match(r"^Art\.?\s*(\d+)[ºo]?", linha, re.IGNORECASE)
        if artigo_match:
            artigo_atual = {
                "tipo": "artigo",
                "numero": artigo_match.group(1),
                "texto": linha,
                "paragrafos": [],
                "incisos": []
            }
            unidades.append(artigo_atual)
            continue

        # ---------------------
        # Detecta Parágrafos
        # - § 1º, § 2º, ...
        # - Parágrafo único
        # ---------------------
        par_match = re.match(r"^§\s*(\d+)[ºo]?", linha)
        unico_match = re.match(r"^Parágrafo único", linha, flags=re.IGNORECASE)

        if (par_match or unico_match) and artigo_atual:
            numero = par_match.group(1) if par_match else "único"
            paragrafo = {
                "tipo": "paragrafo",
                "numero": numero,
                "texto": linha,
                "incisos": []
            }
            artigo_atual["paragrafos"].append(paragrafo)
            continue

        # ---------------------
        # Detecta Incisos (I - … XIX - …)
        # ---------------------
        inciso_match = re.match(
            r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX)\s*[-–]",  # noqa
            linha
        )
        if inciso_match:
            inciso = {
                "tipo": "inciso",
                "numero": inciso_match.group(1),
                "texto": linha
            }
            if artigo_atual:
                if artigo_atual["paragrafos"]:
                    artigo_atual["paragrafos"][-1]["incisos"].append(inciso)
                else:
                    artigo_atual["incisos"].append(inciso)
            continue

        # ---------------------
        # Texto adicional (continuação)
        # ---------------------
        if artigo_atual:
            if artigo_atual["paragrafos"]:
                artigo_atual["paragrafos"][-1]["texto"] += " " + linha
            else:
                artigo_atual["texto"] += " " + linha

    # Metadados
    for art in unidades:
        art["lei_nome"] = lei_nome
        art["lei_numero"] = lei_numero
        art["url"] = url

    return unidades

# ---------------------------
# Loop principal
# ---------------------------

for lei in LEIS:
    try:
        print(f"Baixando: {lei['nome_popular']}")
        r = requests.get(lei["url"], timeout=20, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")

        # Remove tags desnecessárias
        for tag in soup(["script", "style", "head", "title", "meta"]):
            tag.decompose()

        print(f'Extraindo: {lei['nome_popular']}')
        unidades = extrair_estrutura(
            soup, lei["nome_popular"],
            lei["numero"],
            lei["url"]
            )

        # Salvar JSON
        arquivo_json = os.path.join(PASTA_JSON, f"{lei['nome_popular'].replace(' ','_')}.json") #noqa
        with open(arquivo_json, "w", encoding="utf-8") as f:
            json.dump(unidades, f, ensure_ascii=False, indent=4)

        # Salvar TXT
        texto_limpo = "\n".join([u["texto"] for u in unidades])
        arquivo_txt = os.path.join(PASTA_TXT, f"{lei['nome_popular'].replace(' ','_')}.txt") #noqa
        with open(arquivo_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        print(f"Concluído: {lei['nome_popular']}")
    except Exception as e:
        print(f"Erro ao processar {lei['nome_popular']}: {e}")
