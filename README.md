# Data Migration Automation ITS Analysis

Repositório com scripts e materiais de análise estatística sobre a associação entre a implantação gradual de uma ferramenta interna de automação da transformação de dados, com apoio a verificações e validações operacionais, e o desempenho operacional de migrações de dados em contexto LegalTech.

## Objetivo

Organizar, de modo reproduzível, os códigos, a base agregada e os artefatos analíticos utilizados para avaliar indicadores operacionais antes e depois da implantação gradual da automação, com comparações pré e pós-implantação e modelagem por série temporal interrompida (ITS).

## Link do GitHub

<https://github.com/HaustinStelmastchukVieira/data-migration-automation-its>

## Conteúdo principal

- scripts de análise em Python;
- figuras e tabelas geradas a partir da análise;
- base de dados agregada em nível mensal, sem microdados;
- rotinas para comparações pré e pós-implantação;
- modelos de série temporal interrompida (ITS) com erros-padrão HAC de Newey-West.

## Estrutura

```text
.
├── data/
├── .gitignore
├── Makefile
├── requirements.txt
├── src/
├── tests/
└── README.md
```

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_all
```

Com `make`:

```bash
make setup
make run
```

As saídas são geradas automaticamente em `outputs/tables/` e `outputs/figures/`.

## Figuras produzidas

- `qq_volume_panel.png`: painel Q-Q para o volume mensal pré e pós-implantação;
- `pre_post_72h_point_ci.png`: proporção agregada concluída em `≤ 72 horas` com IC95%;
- `its_volume_publication.png`: ITS linear para volume mensal;
- `its_72h_publication.png`: ITS logístico para proporção em `≤ 72 horas`.

## Testes

```bash
pytest
```

Ou:

```bash
make test
```

## Observações

O repositório não contém dados sensíveis, identificáveis ou microdados operacionais.

A base disponibilizada está agregada por mês, em consonância com o desenho analítico adotado no estudo.

O material foi organizado de modo reprodutível para permitir a replicação das análises estatísticas, tabelas e figuras reportadas.
