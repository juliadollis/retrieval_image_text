````markdown
# ELIP-C igual ao paper

Este diretório contém uma reimplementação do ELIP-C (variante baseada em CLIP) o mais fiel possível ao paper original.  

A implementação inclui:

- Backbone CLIP congelado
- Prompts visuais inseridos no início do ViT (VPT-shallow)
- Loss InfoNCE simétrica texto–imagem
- Hard sample mining global Texto→Imagem
- Re-ranking em duas etapas (baseline CLIP e depois ELIP-C guiado por prompt)
- Suporte para treino rápido em subsets pequenos (cc3m e Pokémon)
- Configuração para treino “oficial” com subset de 6M do DataCompDR-12M
- Scripts para push de checkpoints para o Hugging Face Hub

O número de épocas não está especificado diretamente no paper que usamos como base. Por isso, o campo epochs é parametrizado nas configs (padrão 1, podendo ser aumentado).

------------------------------------------------
## Estrutura do diretório

A estrutura esperada é:

- elip_igualpaper/
  - __init__.py
  - configs/
    - test_cc3m.yaml
    - full_datacomp.yaml
    - test_pokemon.yaml
  - models/
    - __init__.py
    - mapper.py
    - backbone_prompted_clip.py
  - utils/
    - __init__.py
    - data.py
    - hardmine.py
  - training/
    - __init__.py
    - train_prompted_clip.py
  - infer/
    - __init__.py
    - infer_rerank_prompted_clip.py
  - scripts/
    - push_to_hf.py

------------------------------------------------
## Instalação de dependências e criação da estrutura

A partir de um diretório como /workspace:

```bash
cd /workspace
mkdir -p elip_igualpaper/{configs,models,utils,training,infer,scripts}
touch elip_igualpaper/__init__.py elip_igualpaper/models/__init__.py elip_igualpaper/utils/__init__.py elip_igualpaper/training/__init__.py elip_igualpaper/infer/__init__.py

pip install -U "torch>=2.6" transformers datasets pillow sentencepiece faiss-cpu huggingface_hub safetensors tqdm webdataset
````

Se houver problemas de compatibilidade entre torch e torchvision, veja a seção de troubleshooting no final deste README.

---

## Configurações principais (configs)

### Config de teste rápido com cc3m (test_cc3m.yaml)

Teste rápido com 1000 amostras do dataset pixparse/cc3m-wds, mantendo o pipeline completo ELIP-C.

```bash
cat > elip_igualpaper/configs/test_cc3m.yaml <<'YAML'
model_name: openai/clip-vit-base-patch32
img_size: 224
prompt_tokens: 12
seed: 3407
output_root: /workspace/elip_igualpaper/out_test_cc3m

dataset:
  name: pixparse/cc3m-wds
  split: train
  image_column: jpg
  text_column: txt
  text_is_list: false

max_train_examples: 1000
hardmine_subset: 1000
batch_size: 64
num_workers: 4
epochs: 1
lr: 5.0e-5
weight_decay: 0.01
fp16: true
grad_accum_steps: 1

rerank_k:
  coco: 100
  flickr: 100
  occluded_coco: 500
  imagenet_r: 1000

eval_split:
  name: pixparse/cc3m-wds
  split: train
  image_column: jpg
  text_column: txt
  text_is_list: false
  max_eval_examples: 500
YAML
```

### Config “oficial” para DataCompDR-12M (full_datacomp.yaml)

Subset aleatório de 6M amostras de apple/DataCompDR-12M-bf16, batch size 40, LR 1e-3 e avaliação em COCO em inglês, como descrito no paper.

```bash
cat > /workspace/elip_igualpaper/configs/full_datacomp.yaml <<'YAML'
model_name: openai/clip-vit-base-patch32
img_size: 224
prompt_tokens: 12
seed: 3407
output_root: /workspace/elip_igualpaper/out_full_datacomp

dataset:
  name: apple/DataCompDR-12M-bf16
  split: train
  image_column: image
  text_column: caption
  text_is_list: false

max_train_examples: 6000000
hardmine_subset: 6000000
batch_size: 40
num_workers: 4
epochs: 1
lr: 1.0e-3
weight_decay: 0.01
fp16: true
grad_accum_steps: 1

rerank_k:
  coco: 100
  flickr: 100
  occluded_coco: 500
  imagenet_r: 1000

eval_split:
  name: flax-community/coco2017-caption
  split: validation
  image_column: image
  text_column: captions
  text_is_list: true
  max_eval_examples: 5000
YAML
```

Para conferir se a config está correta:

```bash
python - <<'PY'
import yaml
y = yaml.safe_load(open("/workspace/elip_igualpaper/configs/full_datacomp.yaml"))
print("max_train_examples:", y["max_train_examples"])
print("hardmine_subset:", y["hardmine_subset"])
print("eval_split:", y["eval_split"]["name"], y["eval_split"]["split"])
PY
```

### Config de teste com Pokémon (test_pokemon.yaml)

Teste menor com reach-vb/pokemon-blip-captions, reutilizando exatamente o mesmo pipeline.

```bash
cat > /workspace/elip_igualpaper/configs/test_pokemon.yaml <<'YAML'
model_name: openai/clip-vit-base-patch32
img_size: 224
prompt_tokens: 12
seed: 3407
output_root: /workspace/elip_igualpaper/out_test_pokemon

dataset:
  name: reach-vb/pokemon-blip-captions
  split: train
  image_column: image
  text_column: caption
  text_is_list: false

max_train_examples: 1000
hardmine_subset: 1000
batch_size: 64
num_workers: 4
epochs: 1
lr: 5.0e-5
weight_decay: 0.01
fp16: true
grad_accum_steps: 1

rerank_k:
  coco: 100
  flickr: 100
  occluded_coco: 500
  imagenet_r: 1000

eval_split:
  name: reach-vb/pokemon-blip-captions
  split: train
  image_column: image
  text_column: caption
  text_is_list: false
  max_eval_examples: 500
YAML
```

---

## Componentes principais

### models/mapper.py

Define o MLPMapper, que mapeia embeddings de texto (d_proj) para tokens de prompt visuais no espaço do ViT (d_model × num_tokens).

Principais pontos:

* Entrada: vetor de texto da cabeça de projeção do CLIP (d_proj)
* Saída: prompt visual com num_tokens tokens no espaço de embedding da visão (d_model)
* Arquitetura: MLP com duas camadas intermediárias e ativação GELU

### models/backbone_prompted_clip.py

Define a classe PromptedCLIP, que:

* Carrega o CLIP da biblioteca transformers usando safetensors
* Congela todos os parâmetros do backbone
* Oferece três métodos principais:

  * encode_text_hidden: embeddings de texto normalizados
  * encode_image_base: embeddings de imagem padrão (CLIP original)
  * encode_image_guided: embeddings de imagem usando tokens de prompt inseridos entre CLS e patches (VPT-shallow)

Essa implementação mantém o CLIP base intacto e aplica as modificações apenas no pipeline de visão via prompts.

### utils/data.py

Responsável por carregar pares imagem–texto a partir de datasets do Hugging Face. Possui:

* Conversão segura para RGB
* Tratamento de campos texto como lista (text_is_list)
* Função _infer_columns para autodetectar nomes de colunas de imagem e texto se os nomes passados não existirem (útil para datasets pequenos com convenções diferentes)

### utils/hardmine.py

Implementa:

* Cálculo de embeddings base para texto e imagem (encode_text_hidden e encode_image_base)
* Hard sample mining global Texto→Imagem:

  * Cria índice FAISS no espaço de imagens
  * Efetua busca com embeddings de texto
  * Para cada texto, seleciona as imagens mais difíceis (mais similares), gerando batches compostos por uma âncora e vizinhos difíceis

A mineração é Texto→Imagem, alinhada com a descrição da Seção de hard sample mining global do paper.

### training/train_prompted_clip.py

Script principal de treino:

* Lê a config definida em ELIP_CFG
* Configura o dispositivo, semente e diretório de saída
* Carrega o backbone PromptedCLIP e congela o CLIP
* Inicializa o MLPMapper e a InfoNCELoss simétrica
* Carrega o dataset e aplica hard sample mining global T→I
* Treina por epochs, salvando checkpoints mapper_epochX.safetensors e mapper_last.safetensors usando safetensors

A loss InfoNCE é simétrica (texto e imagem, com cross-entropy em ambas as direções).

### infer/infer_rerank_prompted_clip.py

Script de inferência e re-ranking:

* Carrega a mesma config via ELIP_CFG
* Carrega o backbone CLIP congelado e o MLPMapper a partir do checkpoint
* Extrai embeddings base de imagem e texto
* Avalia o baseline CLIP (sem prompts) com Recall@1, 5, 10, 50, MRR e nDCG@10
* Executa re-ranking:

  * Define rerank_k com base no nome do dataset: coco, flickr, occluded_coco ou imagenet_r
  * Reranqueia o top-k das imagens usando embeddings guiados por prompt
  * Calcula novamente as métricas para o ranking reranqueado

Imprime na tela as métricas Base e Rerank para comparação.

### scripts/push_to_hf.py

Script para empacotar e enviar checkpoints para o Hugging Face Hub:

* Lê ELIP_CFG para encontrar output_root
* Localiza mapper_last.safetensors (e, opcionalmente, mapper_epochX.safetensors)
* Cria o repositório HF (se não existir)
* Sobe:

  * mapper_last.safetensors
  * Todos os mapper_epoch*.safetensors
  * config.yaml (a config usada)
  * README.md com um mini card contendo informações do modelo base e quantidade de tokens de prompt

O script espera que HF_REPO_ID e HF_TOKEN estejam definidos, ou que o token esteja configurado no ambiente via hf auth login.

---

## Como rodar

### Teste rápido com cc3m (1000 amostras)

```bash
export PYTHONPATH=/workspace/elip_igualpaper:$PYTHONPATH
export ELIP_CFG=/workspace/elip_igualpaper/configs/test_cc3m.yaml
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m elip_igualpaper.training.train_prompted_clip
python -m elip_igualpaper.infer.infer_rerank_prompted_clip
```

### Teste rápido com Pokémon

```bash
export PYTHONPATH=/workspace/elip_igualpaper:$PYTHONPATH
export ELIP_CFG=/workspace/elip_igualpaper/configs/test_pokemon.yaml
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m elip_igualpaper.training.train_prompted_clip
python -m elip_igualpaper.infer.infer_rerank_prompted_clip
```

### Treino “oficial” com DataCompDR-12M (subset 6M) e avaliação em COCO

```bash
export PYTHONPATH=/workspace/elip_igualpaper:$PYTHONPATH
export ELIP_CFG=/workspace/elip_igualpaper/configs/full_datacomp.yaml
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m elip_igualpaper.training.train_prompted_clip
python -m elip_igualpaper.infer.infer_rerank_prompted_clip
```

---

## Push para o Hugging Face Hub

Exemplo de push do checkpoint de teste Pokémon:

```bash
hf auth login
export HF_REPO_ID=seu-usuario/elip-c-pokemon-test
export HF_TOKEN=$(huggingface-cli whoami -t)
export PYTHONPATH=/workspace/elip_igualpaper:$PYTHONPATH
export ELIP_CFG=/workspace/elip_igualpaper/configs/test_pokemon.yaml

python -m elip_igualpaper.scripts.push_to_hf
```

Exemplo para o modelo treinado em DataCompDR-12M:

```bash
hf auth login
export HF_REPO_ID=seu-usuario/elip-c-datacomp12m
export PYTHONPATH=/workspace/elip_igualpaper:$PYTHONPATH
export ELIP_CFG=/workspace/elip_igualpaper/configs/full_datacomp.yaml
python -m elip_igualpaper.scripts.push_to_hf
```

Se preferir, defina HF_TOKEN manualmente:

```bash
export HF_TOKEN=SEU_TOKEN_AQUI
export HF_REPO_ID=seu-usuario/elip-c-datacomp12m
python -m elip_igualpaper.scripts.push_to_hf
```

---

## Onde, como e por que esta implementação difere do CLIP “puro”

1. Uso de safetensors
   Todos os checkpoints (mapper e CLIP) usam safetensors, evitando problemas de segurança associados a torch.load e seguindo as melhores práticas atuais.

2. Hard sample mining global Texto→Imagem
   Em vez de mineração local ou imagem→texto, o hard mining é feito Texto→Imagem, com FAISS no espaço de imagens consultado por embeddings de texto, alinhado à descrição do paper.

3. Arquitetura ELIP-C replicada
   Backbone CLIP permanece congelado.
   Prompts visuais são inseridos no início do ViT (logo após o token CLS).
   Um MLPMapper mapeia embeddings de texto para tokens de prompt visual.
   A loss é InfoNCE simétrica.
   O re-ranking é feito em duas etapas: baseline CLIP e depois guided CLIP com prompts.

4. Extra
   Há logs detalhados com tqdm e prints em todas as fases do pipeline: carregamento, hard mining, treino, inferência e re-ranking.

---

## Troubleshooting: problemas com PyTorch e torchvision

Se aparecer erros relacionados a torchvision (por exemplo, falhas em ops como nms ou incompatibilidade de CUDA), é provável que as versões de torch e torchvision estejam desencontradas no ambiente.

Passos sugeridos:

1. Conferir versões atuais

```bash
python - <<'PY'
import sys, torch
print("python:", sys.version)
print("torch:", getattr(torch, "__version__", None))
print("cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision: ERRO ->", repr(e))
PY
```

2. Ajustar a pilha torch / torchvision

```bash
pip install -U --no-cache-dir pip setuptools wheel

pip uninstall -y torchvision torch torchaudio || true

pip install -U --no-cache-dir "transformers>=4.46.3" "huggingface_hub>=0.25.2" "safetensors>=0.4.5" pillow faiss-cpu

pip install -U --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  || pip install -U --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

3. Validar

```bash
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda.is_available:", torch.cuda.is_available())
PY
```

4. Rodar novamente o treino e a inferência

```bash
export PYTHONPATH=/workspace/elip_igualpaper:$PYTHONPATH
export ELIP_CFG=/workspace/elip_igualpaper/configs/test_cc3m.yaml
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m elip_igualpaper.training.train_prompted_clip
python -m elip_igualpaper.infer.infer_rerank_prompted_clip
```

Se ainda ocorrerem erros, revise as versões instaladas de torch, torchvision e torchaudio, e ajuste conforme a GPU e a versão de CUDA disponíveis no container.

```
```
