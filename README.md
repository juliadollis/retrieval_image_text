

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/1d1db614-5fd7-4ee0-82eb-1d92f672f882" />



---

## O que é Retrieval Imagem-Texto?

Retrieval imagem-texto é a tarefa de buscar imagens relevantes a partir de um texto (texto → imagem) ou buscar textos relevantes a partir de uma imagem (imagem → texto). A ideia central é que um modelo multimodal aprende a representar imagens e textos no mesmo espaço vetorial, onde itens semanticamente semelhantes ficam próximos.

Isso permite que uma consulta como “um cachorro correndo na praia” retorne automaticamente as imagens mais próximas desse conceito. Da mesma forma, fornecer uma imagem permite recuperar descrições ou legendas que melhor representam seu conteúdo.

Retrieval é uma das técnicas fundamentais de visão-linguagem e serve como base para diversos sistemas modernos, como RAG multimodal, motores de busca visuais, geração guiada por imagem e organização de grandes acervos multimodais.

---

## Para que usamos Retrieval?

Busca por imagens
Usado para encontrar rapidamente, dentro de milhares ou milhões de imagens, aquelas que são mais relevantes para um texto. Aplicações incluem mecanismos de busca, bancos de arte, fotografia, e-commerces e catálogos digitais.

Organização de acervos multimodais
Auxilia na criação de índices, na recomendação visual e na detecção de similaridade.

RAG multimodal
Em modelos que entendem imagens e textos, recuperar imagens relevantes melhora profundamente a qualidade das respostas. Exemplo: perguntar sobre uma obra de arte e recuperar quadros correlatos para análise.

Sistemas interativos
Aplicações diversas, como buscar receitas pela foto, encontrar documentos escaneados semelhantes ou localizar produtos no varejo a partir de uma imagem.

Base para modelos generativos
Muitos modelos usam retrieval para encontrar pares semelhantes, auxiliar no condicionamento ou melhorar datasets utilizados em pré-treinamentos.

---

## Quais modelos podemos usar para retrieval e por quê?

Modelos CLIP (OpenAI)
Simples, rápidos e amplamente usados. Boas opções para ensino e experimentação.
Limitados em textos longos e idiomas fora do inglês.

Modelos CLIP da comunidade (OpenCLIP, LAION)
openai/clip-vit-base-patch32
openai/clip-vit-large-patch14
laion/CLIP-ViT-H-14-laion2B
Modelos maiores, treinados em datasets massivos, com desempenho superior ao CLIP original.

Modelos SigLIP (Google)
google/siglip-so400m-patch14-384
google/siglip-base-patch16-224
Mais estáveis, geralmente melhores que CLIP e fortes em zero-shot.

SigLIP 2
Suporte nativo a múltiplos idiomas, incluindo português. Desempenho excelente para tasks multilíngue.

Modelos fine-tunados em domínio específico
Exemplos: flickr30k-pt-br-5ep, wikiart-ft
Adaptam embeddings ao domínio, melhoram recall@K e superam facilmente modelos base.

Encoders híbridos (BLIP, BLIP-2, LLaVA, Qwen-VL, Janus-Pro)
Possuem encoders fortes, podem ser usados para retrieval mesmo não sendo o objetivo principal.
São ideais para RAG multimodal, análise visual e geração de respostas multimodais.
Desempenho geralmente inferior a CLIP/SigLIP para retrieval puro.

---

## Papers essenciais para começar

CLIP (2021) – OpenAI
Learning Transferable Visual Models From Natural Language Supervision
Introduziu o treinamento contraste imagem-texto em larga escala e estabeleceu a base moderna do campo.

ALIGN (2021) – Google
Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
Explora pré-treinamento massivo utilizando bilhões de pares imagem-texto ruidosos.

SigLIP (2024) – Google
Scaling Vision-Language Models with Sigmoid Loss
Uma evolução direta do CLIP com uma função de perda mais estável. É o baseline mais forte em diversos benchmarks modernos.

---

## Mais papers importantes para aprofundar

VSE++ (2018) – Hard negatives matter
Mostra a importância de amostras difíceis no treinamento contraste.

DeCLIP (2021) – Debiasing contrastive representations
Introduz regularização, distorção multimodal e pseudo-rótulos.

FLIP (2022) – Data augmentations para CLIP
Mostra como augmentations agressivos melhoram o desempenho.

Papers importantes de modelos de RAG multimodal
BLIP-2 (2023), LLaVA (2023), Qwen-VL (2024), Janus-Pro (2024)

Benchmarks e datasets
MS-COCO
Flickr30k
XTD-10
Crossmodal-3600
Winoground

---

## Como um modelo de Retrieval funciona

O princípio central é representar imagens e textos no mesmo espaço vetorial. Tudo se resume a converter imagem e texto em vetores e comparar esses vetores usando uma função de similaridade.

Os componentes fundamentais são:

Encoder de imagem
Gera um vetor a partir da imagem. Normalmente utiliza um Vision Transformer (ViT).

Encoder de texto
Gera um vetor a partir do texto usando um Transformer semelhante ao BERT.

Espaço vetorial compartilhado
Os dois encoders são treinados para mapear suas entradas para vetores comparáveis.

Perda contrastiva
Pares corretos são aproximados.
Pares incorretos são afastados.

O treinamento é feito com milhões ou bilhões de pares imagem-texto.

---

## Arquiteturas principais: CLIP e SigLIP

CLIP
Usa perda InfoNCE com softmax.
Extremamente difundido.
Treinado com 400 milhões de pares.
Sofre um pouco com multilinguismo.

SigLIP
Substitui softmax por sigmoid loss.
Mais estável e robusto.
Desempenho superior ao CLIP em vários cenários.
SigLIP 2 adiciona multilinguismo, com suporte avançado para português.

---

## Pipeline completo de retrieval

1. Carregar dataset
   Dataset precisa ter uma coluna image e outra text.

2. Extrair embeddings
   Gera vetores de tamanho fixo (512, 768, 1024).

3. Construir índice FAISS
   Indexa embeddings das imagens (ou textos).

4. Consultar o sistema
   texto → imagem
   imagem → texto

5. Avaliar
   Uso das métricas Recall@K, MRR e nDCG@K.

6. Visualizar resultados
   Mostrar imagens recuperadas, ranks e comparação entre modelos.

---

## Métricas de Avaliação

Recall@K
Mede se o item correto aparece entre os K primeiros resultados.

MRR
Avalia a posição exata do item correto.
Mais sensível à ordenação do ranking.

nDCG@K
Avalia qualidade completa do ranking.
Recompensa itens bem posicionados no topo.

Similaridade média
Diagnóstico útil para ver estabilidade do embedding.

Avaliação cruzada
Testes em duas direções:
texto → imagem
imagem → texto

---

## Como interpretar resultados

Exemplo:

Modelo A
Recall@1 = 0.20
Recall@5 = 0.45
MRR = 0.30
nDCG@10 = 0.35

Modelo B
Recall@1 = 0.32
Recall@5 = 0.62
MRR = 0.46
nDCG@10 = 0.52

Conclusão:
O modelo B melhora todas as métricas, ordena melhor, encontra resultados mais cedo e tem ranking superior.






### Terminamos a "Teoria"... agora começa a prática!

Depois de entender o que é Retrieval Imagem–Texto, chegou a hora de trabalhar com os seus dados, avaliar modelos e decidir os próximos passos.  
Esta seção foi organizada como um guia sequência-de-decisão para te ajudar a ir do zero até a implementação completa.

---

## OK, já sei o que é Retrieval, tenho meus dados. Como eu começo?

### 1. Organize seus dados
Neste repositório trabalhamos exclusivamente com *datasets do Hugging Face*.  
Certifique-se de ter:

- Uma coluna contendo imagens (`image`)
- Uma coluna contendo textos/legendas (`text` ou `caption`)

Existem dois cenários:

**a. Seus dados são próprios (locais)**  
Prepare um script para converter seus arquivos em um dataset do Hugging Face.  
Qualquer estrutura simples como:

```

{"image": <PIL.Image>, "text": "<sua legenda>"}

```

já é suficiente.

**b. Você está usando um dataset público da HF**  
Perfeito. Apenas confira quais são as colunas de imagem e texto e adapte o script de inferência/fine-tuning.

---

### 2. Faça a primeira avaliação sem fine-tuning (zero-shot)

Antes de treinar qualquer coisa, avalie a qualidade do modelo base no seu domínio.  
Use este checklist:

- Carregar o modelo e o processor.
- Extrair embeddings das imagens do **split de teste**.
- Extrair embeddings dos textos do mesmo split (mesmo índice).
- Calcular similaridade coseno entre textos e imagens.
- Para cada texto, rankear imagens e medir:
  - Recall@K (1, 5, 10)
  - MRR
  - nDCG@K

Isso responde perguntas fundamentais:

- O modelo base entende minimamente o seu domínio?
- Suas legendas estão boas ou muito genéricas?
- Vale a pena investir em fine-tuning ou o baseline já é suficiente?

- Olhe a seção # Como eu analiso o desempenho dos meus dados nos modelos?

---

### 3. Decida seu próximo passo

**a. O modelo base foi bem**  
Ótimo! Continue usando zero-shot. Documente suas métricas e siga para o desenvolvimento da sua aplicação.

**b. O modelo base teve desempenho mediano**  
Faça fine-tuning com seus dados (arte, médico, jurídico, moda, etc.).

**c. Você tem muitos dados (milhões de pares)**  
Considere abordagens avançadas como ELIP, hard sample mining, prompting e reranking.

**d. Você tem objetivos específicos**  
Pode explorar:
- RAG multimodal
- Recomendação visual
- Buscadores inteligentes
- Geração condicionada por imagens recuperadas

---

# Como eu analiso o desempenho dos meus dados nos modelos?

### 1. Usando código local
Na pasta `InferenciaEMetricas/` você encontra:

- `inferencias_clip.py`
- `inferencias_siglip.py`

Nesses scripts você só precisa alterar:
- Nome do dataset
- Split
- Nome do modelo

---

### 2. Usando a Demonstração Automática
Existe uma demonstração que:

- Carrega seu dataset HuggingFace (público)
- Extrai embeddings
- Avalia modelos CLIP e SigLIP automaticamente
- Calcula todas as métricas
- Gera interpretação dos resultados

Você só precisa fornecer:
- Caminho do dataset
- Split
- Nome das colunas

**Link da Demonstração:**  
(em breve no README)

---

# Analisei meus dados. Já sei o que quero fazer!

Agora você pode escolher entre Fine-Tuning, Inferência, Métricas ou até replicar o ELIP.

---

# Implementações do Repositório

## Fine Tuning

Nesta etapa você adapta o modelo base ao seu domínio.

Scripts estão em: `FineTuning/`

Você deve:
- Definir o modelo base (`MODEL_NAME`)
- Definir o dataset (`DATASET_NAME`)
- Ajustar:
  - batch size
  - número de épocas
  - learning rate

Sempre que fizer um novo experimento, lembre de:
- Atualizar o caminho do modelo em `MODEL_NAME`
- Atualizar `HUB_MODEL_ID` (nome do modelo salvo no Hub)

Após o treino:
- O modelo é salvo no diretório de saída
- Opcionalmente enviado para o Hugging Face Hub

---

## Inferência e Métricas

Scripts em: `InferenciaEMetricas/`

Avaliando o modelo:
- Aponte para o modelo correto:
  - Base (`google/siglip-base-patch16-256-multilingual`)
  - Fine-tuned (`turing552/siglip-wikiart-5ep`)
- Carregue o mesmo split de teste
- Calcule:
  - Recall@1, Recall@5, Recall@10
  - MRR
  - nDCG@10

Checklist ao trocar o modelo:
1. Atualizar `MODEL_NAME`
2. Carregar o processor correspondente
3. Garantir que o split de teste é o mesmo do fine-tuning
4. Conferir colunas de imagem e texto

Registrar os resultados no README facilita comparar:
- Base vs fine-tuned
- Diferentes domínios
- Diferentes datasets

---

# Requisitos para Replicar ELIP

ELIP é pesado e exige hardware e datasets grandes. Se quiser algo fiel ao paper, siga esta estrutura:

```

ELIP/
original/   -> versão mais próxima do paper
wikiart/    -> adaptação para domínio de arte
PTBR/       -> adaptação para datasets em português

```

### Requisitos

1. **Hardware**
   - GPU com no mínimo 32 GB de VRAM  
   - SSD com espaço suficiente para datasets grandes

2. **Dataset do paper**
   - `apple/DataCompDR-12M`
   - Requer ~1 TB

3. **Dataset WeiChow/cc3m**
   - Requer ~800 GB

4. **Datasets menores**
   - `Artificio/WikiArt`
   - `laicsiifes/flickr8k-pt-br`
   - Requerem ~250 GB

Resumo prático:
- Paper original → 1 TB  
- CC3M → 800 GB  
- Experimentos menores → 250 GB

---

# Agora sim: vamos colocar em prática!

## Temos 3 demonstrações prontas para usar Retrieval Imagem–Texto

---

### 1. Demo: Retrieval de Moda

Demonstração online:  
https://e4f8dadade2c8014aa.gradio.live/

Execute localmente:
```

cd Demonstrações
python app_retrieval_moda.py

```

Você pode:
- Usar seu próprio dataset
- Usar seu modelo fine-tuned
- Usar o modelo base

Basta editar:
- Nome do dataset
- Split
- Colunas
- Modelo

Observação:  
As demonstrações usam inglês como idioma padrão, mas você pode adaptar facilmente para português.

```

---

