"""Geração de relatório exploratório em Markdown a partir do dataset bruto."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

DATE_FORMAT = "%Y-%m-%d"
RARE_CLASS_THRESHOLDS = (5, 10, 20, 50)


def compute_eda_stats(raw_path: Path) -> dict:
    """Calcula estatísticas exploratórias sem carregar o CSV inteiro em memória."""
    total_rows = 0
    header: list[str] | None = None
    missing: dict[str, int] = {}
    duplicate_titles = 0
    seen_titles: set[str] = set()
    category_counts: Counter[str] = Counter()
    title_lengths: list[int] = []
    valid_dates = 0
    first_date: datetime | None = None
    last_date: datetime | None = None

    with open(raw_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        missing = {column: 0 for column in header}

        for row in reader:
            total_rows += 1

            for column in header:
                value = (row.get(column) or "").strip()
                if not value:
                    missing[column] += 1

            title = (row.get("title") or "").strip()
            if title:
                title_lengths.append(len(title))
                if title in seen_titles:
                    duplicate_titles += 1
                else:
                    seen_titles.add(title)

            category = (row.get("category") or "").strip()
            if category:
                category_counts[category] += 1

            date_str = (row.get("date") or "").strip()
            parsed_date = parse_valid_date(date_str)
            if parsed_date is not None:
                valid_dates += 1
                first_date = parsed_date if first_date is None else min(first_date, parsed_date)
                last_date = parsed_date if last_date is None else max(last_date, parsed_date)

    mean_len, median_len, min_len, max_len = compute_title_length_summary(title_lengths)

    rare_thresholds = {}
    for threshold in RARE_CLASS_THRESHOLDS:
        rare_classes = {category: count for category, count in category_counts.items() if count < threshold}
        rare_thresholds[threshold] = {
            "classes": len(rare_classes),
            "samples": sum(rare_classes.values()),
        }

    least_frequent_classes = sorted(category_counts.items(), key=lambda item: (item[1], item[0]))[:10]

    return {
        "header": header,
        "total_rows": total_rows,
        "missing": missing,
        "duplicate_titles": duplicate_titles,
        "category_counts": category_counts,
        "mean_len": mean_len,
        "median_len": median_len,
        "min_len": min_len,
        "max_len": max_len,
        "first_date": first_date.strftime(DATE_FORMAT) if first_date else "N/A",
        "last_date": last_date.strftime(DATE_FORMAT) if last_date else "N/A",
        "valid_dates": valid_dates,
        "rare_thresholds": rare_thresholds,
        "least_frequent_classes": least_frequent_classes,
    }


def parse_valid_date(date_str: str) -> datetime | None:
    """Converte a data do CSV para `datetime` quando o valor estiver no formato esperado."""
    if not date_str:
        return None

    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except ValueError:
        return None


def compute_title_length_summary(title_lengths: list[int]) -> tuple[float, float, int, int]:
    """Resume média, mediana e extremos do tamanho dos títulos."""
    title_lengths.sort()
    if not title_lengths:
        return 0, 0, 0, 0

    total_titles = len(title_lengths)
    mean_len = sum(title_lengths) / total_titles
    median_len = (title_lengths[(total_titles - 1) // 2] + title_lengths[total_titles // 2]) / 2
    min_len = title_lengths[0]
    max_len = title_lengths[-1]
    return mean_len, median_len, min_len, max_len


def render_markdown(stats: dict) -> str:
    """Gera o relatório em Markdown a partir das estatísticas calculadas."""
    total_rows = stats["total_rows"]
    header = stats["header"]
    missing = stats["missing"]
    category_counts = stats["category_counts"]
    duplicate_titles = stats["duplicate_titles"]
    rare_thresholds = stats["rare_thresholds"]

    md: list[str] = []
    md.append("# Relatório de Análise Exploratória de Dados\n")
    md.append("- **Dataset analisado:** `data/raw/articles.csv`")
    md.append(f"- **Total de registros lidos:** {total_rows}")
    md.append(f"- **Número de colunas:** {len(header)} ({', '.join(f'`{column}`' for column in header)})\n")

    md.append("## Resumo executivo\n")
    md.append("- A base tem volume suficiente para um classificador supervisionado, mas apresenta desbalanceamento relevante entre categorias.")
    md.append("- A coluna `title` está completa e representa o mesmo dado que a API recebe em produção, por isso foi adotada como entrada principal.")
    md.append("- A coluna `subcategory` tem taxa de ausência alta demais para sustentar o fluxo principal.")
    md.append("- Há classes com pouquíssimas amostras, o que reforça a necessidade de frequência mínima no treino.")
    md.append("- Os achados favorecem um pipeline simples e reproduzível com pré-processamento, `TF-IDF` e seleção por `macro_f1`.\n")

    md.append("## 1. Valores ausentes por coluna\n")
    md.append("| Coluna | Valores ausentes | % ausente |")
    md.append("| :--- | :---: | :---: |")
    for column in header:
        pct = (missing[column] / total_rows) * 100 if total_rows else 0
        md.append(f"| {column} | {missing[column]} | {pct:.2f}% |")
    md.append("\n")
    md.append("Leitura: `title`, `date`, `category` e `link` estão completos, enquanto `subcategory` tem ausência elevada e `text` possui poucos nulos.\n")

    md.append("## 2. Linhas duplicadas com base no título\n")
    dup_pct = (duplicate_titles / total_rows) * 100 if total_rows else 0
    md.append(f"- **Títulos duplicados identificados:** {duplicate_titles} ({dup_pct:.2f}% do total)")
    md.append("- Esse diagnóstico fica registrado como atenção para futuras iterações, especialmente em estratégias de split e deduplicação.\n")

    md.append("## 3. Distribuição das categorias\n")
    md.append("| Categoria | Quantidade | % do total | Representação |")
    md.append("| :--- | :---: | :---: | :--- |")
    for category, count in category_counts.most_common(20):
        pct = (count / total_rows) * 100 if total_rows else 0
        bar = "#" * int(pct / 2) if int(pct / 2) > 0 else "-"
        md.append(f"| {category} | {count} | {pct:.2f}% | {bar} |")
    if len(category_counts) > 20:
        md.append(f"| *Outras ({len(category_counts) - 20} classes)* | - | - | - |")
    md.append("\n")
    md.append("Leitura: as categorias mais frequentes concentram boa parte da base, então a avaliação precisa considerar o impacto do desbalanceamento.\n")

    md.append("## 4. Estatísticas do tamanho dos títulos\n")
    md.append("| Métrica | Valor |")
    md.append("| :--- | :---: |")
    md.append(f"| Tamanho médio | {stats['mean_len']:.2f} |")
    md.append(f"| Tamanho mediano | {stats['median_len']:.2f} |")
    md.append(f"| Tamanho mínimo | {stats['min_len']} |")
    md.append(f"| Tamanho máximo | {stats['max_len']} |")
    md.append("\n")
    md.append("Leitura: os títulos têm conteúdo suficiente para classificação, mas são curtos o bastante para exigir limpeza consistente e validação mínima na API.\n")

    md.append("## 5. Cobertura temporal\n")
    md.append("| Métrica | Valor |")
    md.append("| :--- | :---: |")
    md.append(f"| Primeira notícia | {stats['first_date']} |")
    md.append(f"| Última notícia | {stats['last_date']} |")
    md.append(f"| Datas válidas | {stats['valid_dates']} |")
    md.append("\n")
    md.append("Leitura: a base cobre um intervalo contínuo entre 2015 e 2017, adequado para o case, embora uma validação temporal seja um próximo passo natural.\n")

    md.append("## 6. Classes raras e impacto na modelagem\n")
    md.append("| Limiar | Classes abaixo do limiar | Amostras afetadas |")
    md.append("| :--- | :---: | :---: |")
    for threshold, summary in rare_thresholds.items():
        md.append(f"| < {threshold} | {summary['classes']} | {summary['samples']} |")
    md.append("\n")
    md.append("- **Classes mais raras:** " + ", ".join(f"`{category}` ({count})" for category, count in stats["least_frequent_classes"]))
    md.append("- Esse diagnóstico sustenta o uso de frequência mínima no treinamento para evitar classes com sinal estatístico muito fraco.\n")

    md.append("## 7. Decisões de modelagem sustentadas pela EDA\n")
    md.append("1. **Uso do título como entrada principal:** `title` está completo e espelha exatamente o campo recebido pela API.")
    md.append("2. **Não utilização de `subcategory` no fluxo principal:** a taxa de ausência é alta demais para tratá-la como variável central.")
    md.append("3. **Avaliação por `macro_f1`:** o desbalanceamento entre categorias torna essa métrica mais representativa do que observar apenas a acurácia.")
    md.append("4. **Frequência mínima por classe:** a presença de várias classes raras justifica o corte adotado no treinamento.")
    md.append("5. **Pipeline textual simples e reproduzível:** a combinação de pré-processamento, `TF-IDF` e classificadores lineares atende ao escopo do case sem aumentar a complexidade de serving.")
    md.append("6. **Validação mínima na API:** títulos excessivamente curtos são recusados para evitar inferências com contexto insuficiente.\n")

    md.append("## 8. Próximos passos naturais\n")
    md.append("- Deduplicar títulos antes da divisão treino/teste.")
    md.append("- Comparar validação aleatória com validação temporal.")
    md.append("- Investigar estratégias específicas para categorias de baixo suporte.")

    return "\n".join(md)


def main() -> None:
    raw_path = Path("data/raw/articles.csv")
    if not raw_path.exists():
        print("Dataset bruto não encontrado.")
        return

    print("Iniciando EDA na base completa...")
    stats = compute_eda_stats(raw_path)
    print(f"Lidos {stats['total_rows']} registros.")

    report_path = Path("reports/eda_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as out:
        out.write(render_markdown(stats))

    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
