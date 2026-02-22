# Ecstrator ITS Analysis

Repositório de análise estatística (pré-pós e série temporal interrompida - ITS) referente ao estudo sobre automação de transformação/validação em migração de dados em contexto LegalTech.

## Escopo
Este repositório contém apenas:
- scripts de análise estatística;
- geração de figuras e tabelas;
- base de dados **agregada mensal** (julho/2020 a junho/2025).

## Confidencialidade
Não contém:
- microdados de migrações;
- identificadores de clientes;
- sistemas de origem identificáveis;
- regras proprietárias do Ecstrator;
- credenciais, tokens ou configurações internas.

## Estrutura dos dados
`data/monthly_aggregates.csv`
- `month`: mês de referência (YYYY-MM)
- `volume_total`: total de migrações concluídas no mês
- `sucesso_72h`: migrações concluídas em ≤72h no mês
 
Detalhes adicionais em `data/README.md`.

## Requisitos
- Python 3.12+
- bibliotecas em `requirements.txt`

## Como executar
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.run_all
```

## Observações
- O Matplotlib usa `outputs/.mplconfig` como cache (configurado em `src/plots.py`) para evitar problemas de permissão em `~/.config`.
