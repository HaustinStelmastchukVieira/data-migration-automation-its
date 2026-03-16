# Data Migration Automation ITS

Repositório de análise estatística referente ao estudo sobre automação da transformação e validação de dados em migrações no contexto LegalTech, com comparações pré e pós-intervenção e modelagem por série temporal interrompida (ITS).

## Escopo

Este repositório contém apenas:
- códigos de preparação da base agregada para análise;
- rotinas de análise estatística;
- geração de tabelas e figuras;
- base de dados agregada mensal, referente ao período de julho de 2020 a junho de 2025.

## Confidencialidade

Este repositório não contém:
- microdados de migrações;
- identificadores de clientes;
- sistemas de origem identificáveis;
- regras proprietárias da ferramenta interna de automação;
- credenciais, tokens ou configurações internas.

## Estrutura dos dados

`data/monthly_aggregates.csv`

Campos principais:
- `month`: mês de referência (`YYYY-MM`);
- `volume_total`: total de migrações concluídas no mês;
- `sucesso_72h`: total de migrações concluídas em até 72 horas no mês.

Informações complementares podem ser descritas em `data/README.md`.

## Requisitos

- Python 3.12+
- dependências listadas em `requirements.txt`

## Como executar

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.run_all
```

## Reprodutibilidade

Os códigos foram organizados de modo reprodutível e versionado, permitindo replicar as análises a partir da base mensal agregada disponibilizada neste repositório.

## Observações

- O Matplotlib usa `outputs/.mplconfig` como diretório de cache, conforme configuração em `src/plots.py`, para evitar problemas de permissão em `~/.config`.
