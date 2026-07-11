import sys
from pathlib import Path

# Ajustar o path para poder importar módulos do diretório src
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.preprocess import limpar_texto, remover_acentos


def test_remover_acentos():
    assert remover_acentos("Olá, sábado magnífico!") == "Ola, sabado magnifico!"
    assert remover_acentos("Atenção à árvore e ao ímã") == "Atencao a arvore e ao ima"


def test_limpar_texto_basico():
    # Deve converter para minúsculas e separar palavras
    assert limpar_texto("Mundo Cão") == ["mundo", "cao"]


def test_limpar_texto_com_stopwords():
    # Artigos e preposições devem ser removidos ("o", "de", "com")
    assert limpar_texto("O carro de corrida com asas") == ["carro", "corrida", "asas"]


def test_limpar_texto_com_pontuacao():
    # Pontuações devem ser desconsideradas
    assert limpar_texto("futebol, gol! gol... esporte?") == ["futebol", "gol", "gol", "esporte"]


def test_limpar_texto_vazio():
    assert limpar_texto("") == []
    assert limpar_texto("   ") == []
    # Apenas stopwords vira lista vazia
    assert limpar_texto("o de com para") == []
