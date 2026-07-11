"""Utilitários de limpeza textual usados no treino e na inferência."""

import re
import unicodedata

# Regex para manter apenas caracteres alfanuméricos e espaços
WORD_PATTERN = re.compile(r"\b[a-z0-9]+\b")

# Conjunto robusto de stopwords em português (normalizadas sem acento e em minúsculas)
PORTUGUESE_STOPWORDS = {
    # Artigos e preposições comuns
    "a", "o", "as", "os", "ao", "aos", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sob", "sobre", "atras", "contra", "entre", "perante", "apos", "ate",
    "pelo", "pela", "pelos", "pelas",
    
    # Pronomes e determinantes
    "eu", "tu", "ele", "ela", "nos", "vos", "eles", "elas", "me", "te", "se", "lhe", "lhes", "nosso",
    "nossa", "nossos", "nossas", "teu", "tua", "teus", "tuas", "meu", "minha", "meus", "minhas",
    "dele", "dela", "deles", "delas", "este", "esta", "estes", "estas", "isto", "esse", "essa",
    "esses", "essas", "isso", "aquele", "aquela", "aqueles", "aquelas", "aquilo", "qual", "quais",
    "quem", "cujo", "cuja", "cujos", "cujas", "onde", "como", "quando", "porque", "aquele",
    
    # Conjunções e advérbios comuns
    "que", "se", "mas", "ou", "porem", "contudo", "todavia", "entretanto", "pois", "porque",
    "entao", "assim", "logo", "embora", "ja", "bem", "mal", "muito", "pouco", "mais", "menos",
    "tao", "tanto", "quase", "apenas", "somente", "so", "sim", "nao", "talvez", "ainda", "tambem",
    
    # Verbos auxiliares e de ligação muito frequentes
    "ser", "sendo", "sido", "fui", "foi", "fomos", "foram", "era", "eram", "seria", "seriam",
    "sou", "es", "e", "somos", "sois", "sao", "seja", "sejam", "fosse", "fossem",
    "estar", "estando", "estado", "estive", "esteve", "estivemos", "estiveram", "estava", "estavam",
    "estou", "esta", "estamos", "estao", "esteja", "estejam", "estivesse", "estivessem",
    "ter", "tendo", "tido", "tenho", "tem", "temos", "tendes", "tem", "tinha", "tinham",
    "tive", "teve", "tivemos", "tiveram", "tenha", "tenham", "tivesse", "tivessem",
    "haver", "ha", "hao", "havia", "haviam", "houve", "houveram", "haja", "hajam"
}


def remover_acentos(texto: str) -> str:
    """Remove os acentos e diacríticos de um texto em português.
    
    Exemplo:
        >>> remover_acentos("Olá, sábado!")
        "Ola, sabado!"
    """
    normalized = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def limpar_texto(texto: str) -> list[str]:
    """Realiza a limpeza completa de um texto:
    1. Converte para minúsculas
    2. Remove acentuação
    3. Tokeniza mantendo apenas termos alfanuméricos
    4. Filtra stopwords
    
    Args:
        texto: A string de texto bruto.
        
    Returns:
        Uma lista de tokens limpos e normalizados.
    """
    if not texto:
        return []

    # O mesmo fluxo é compartilhado entre treino e inferência para evitar divergências.
    texto_min = texto.lower()
    texto_sem_acento = remover_acentos(texto_min)
    tokens = WORD_PATTERN.findall(texto_sem_acento)
    tokens_filtrados = [token for token in tokens if token not in PORTUGUESE_STOPWORDS]

    return tokens_filtrados
