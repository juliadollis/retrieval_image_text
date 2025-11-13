

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

---
## OK, já sei o que é Retrieval, tenho meus dados, como eu começo?

1. Tenha seus dados organizados
   Nesse repositório usamos apenas dados do Hugging Face. Garanta que exista uma coluna para a imagem e uma para o texto.

   a. Seus dados são próprios e estão locais
   Então monte um script para subir seus dados como um dataset no seu repositório do Hugging Face (por exemplo, com colunas `image` e `text` ou `caption`).

   b. É um dataset público do Hugging Face
   Perfeito. Basta verificar quais são as colunas de imagem e texto e adaptar o código para usá-las.

2. Faça a primeira avaliação sem fine-tuning (zero-shot)
   Siga este checklist:

   * Carregar o modelo e o processor.
   * Extrair embeddings das imagens do split de teste.
   * Extrair embeddings dos textos do mesmo split (mesmo índice das imagens).
   * Calcular similaridade coseno entre embeddings de texto e imagem.
   * Para cada texto, rankear as imagens e medir Recall@K, MRR e nDCG@K.

   Isso já responde perguntas importantes:

   * O modelo entende razoavelmente bem o seu domínio?
   * Os textos estão claros ou muito confusos?
   * Vale a pena investir em fine-tuning ou o baseline já é bom?

3. Decida o que você fará em seguida

   a. Seus dados tiveram bons resultados com o modelo base
   Continue utilizando o modelo base (zero-shot) e apenas documente as métricas.

   b. Fine-tuning
   Se os resultados estiverem medianos e você tiver dados suficientes, faça fine-tuning do modelo no seu domínio (por exemplo, arte, médico, jurídico).

   c. Muitos dados
   Se você tem muitos dados (milhões de pares) e recursos de hardware, pode considerar abordagens mais pesadas como ELIP ou variações com hard sample mining, prompting e reranking.

   d. Outras abordagens
   Dependendo do seu objetivo, você pode explorar outras abordagens:
   


# Implementando

## Fine Tuning

Nesta etapa você adapta o modelo base ao seu domínio (arte, médico, jurídico, etc.).

- Os scripts de fine-tuning estão na pasta `FineTuning`.
- Nesses scripts você deve:
  - Definir o nome do modelo base (por exemplo, CLIP ou SigLIP).
  - Definir o nome do dataset do Hugging Face que será usado para treino.
  - Ajustar hiperparâmetros como batch size, número de épocas e learning rate.
- Sempre que mudar de experimento (outro dataset, outro modelo), lembre de:
  - Atualizar o caminho do modelo em `MODEL_NAME` ou variável equivalente.
  - Atualizar o `HUB_MODEL_ID` para salvar cada experimento com um nome diferente no Hugging Face Hub.

Depois do treino, o modelo fine-tunado é salvo na pasta de saída configurada (por exemplo, `./siglip-fwikiart-v1`) e, opcionalmente, enviado para o Hub.


## Inferência e Métricas

Aqui você avalia o modelo (base ou fine-tunado) no seu dataset de teste.

- Os scripts de inferência e cálculo de métricas estão na pasta `InferenciaEMetricas`.
- Nesses scripts você deve:
  - Apontar para o modelo correto:
    - Modelo base (por exemplo, `google/siglip-base-patch16-256-multilingual`), ou
    - Modelo fine-tunado salvo no Hub (por exemplo, `turing552/siglip-wikiart-5ep`).
  - Carregar o mesmo dataset (ou split) que foi usado para teste.
  - Calcular:
    - Recall@K (geralmente K = 1, 5, 10)
    - MRR
    - nDCG@K

Checklist ao mudar o caminho do modelo:

1. Atualizar a variável `MODEL_NAME` (ou equivalente) para o modelo que você quer testar.
2. Garantir que o processor carregado corresponde ao mesmo modelo.
3. Confirmar que o script está usando o split de teste correto e as mesmas colunas de imagem e texto.

Registrar as métricas obtidas em uma tabela no README ajuda a comparar:

- Modelo base vs modelo fine-tunado
- Diferentes datasets
- Diferentes domínios (arte, médico, etc.)


## Requisitos para replicar ELIP

ELIP é um método pesado, que exige preparação de dados e bastante capacidade computacional. A ideia deste repositório é também apontar um caminho para quem quiser reproduzir algo próximo ao paper.

Estrutura sugerida no repositório:

- `ELIP/`
  - `original`   (versão mais próxima do paper, em inglês)
  - `wikiart`    (adaptação para domínio de arte)
  - `PTBR`       (adaptação para datasets em português, como Flickr8k-pt-br)

Requisitos principais:

1. Hardware  
   - GPU com no mínimo 32 GB de VRAM (idealmente mais, se for usar DataCompDR-12M).
   - Espaço em disco adequado para os datasets abaixo.

2. Dataset do paper  
   - `apple/DataCompDR-12M` (ou variante bf16)  
   - Requer cerca de 1 TB de armazenamento.

3. Dataset WeiChow/cc3m  
   - `WeiChow/cc3m`  
   - Requer aproximadamente 800 GB de armazenamento.

4. Datasets menores  
   - `Artificio/WikiArt`  
   - `laicsiifes/flickr8k-pt-br`  
   - Requerem em torno de 250 GB de armazenamento para trabalhar com mais folga (checkpoints, índices, logs, etc.).

Resumo prático:

- Para replicar com o dataset do paper (`apple/DataCompDR-12M`), planeje pelo menos 1 TB de armazenamento.
- Para usar `WeiChow/cc3m`, planeje algo em torno de 800 GB.
- Para experimentos menores com `Artificio/WikiArt` ou `laicsiifes/flickr8k-pt-br`, algo em torno de 250 GB costuma ser suficiente para dados, checkpoints e índices de retrieval.

