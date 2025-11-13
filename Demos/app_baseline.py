import os
import io
import random
import numpy as np
import torch
import faiss
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor, AutoModel
from PIL import Image, ImageFile
from tqdm.auto import tqdm
import gradio as gr

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SEED = 42
MAX_EVAL_ROWS = 1000
TEXT_MAX_LENGTH = 64

MODEL_IDS = [
    "google/siglip-so400m-patch14-384",
    "google/siglip-base-patch16-224",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14"
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def carregar_dataset(dataset_name, splits_str):
    splits = [s.strip() for s in splits_str.split(",") if s.strip()]
    if not splits:
        raise ValueError("Você precisa informar pelo menos um split, por exemplo: train ou train,validation")
    partes = []
    for s in splits:
        parte = load_dataset(dataset_name, split=s)
        partes.append(parte)
    if len(partes) == 1:
        dataset = partes[0]
    else:
        dataset = concatenate_datasets(partes)
    return dataset

def analisar_dataset_base(dataset, text_column):
    total_linhas = len(dataset)
    try:
        valores_unicos = dataset.unique(text_column)
        num_unicos = len(valores_unicos)
    except Exception:
        textos = dataset[text_column]
        num_unicos = len(set(textos))
    return total_linhas, num_unicos

def garantir_pil_rgb(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if "bytes" in img:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if "array" in img:
            arr = np.array(img["array"])
        else:
            raise ValueError("Formato de imagem dict não suportado.")
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise ValueError(f"Tipo de imagem não suportado: {type(img)}")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in [1, 3] and arr.shape[-1] not in [1, 3]:
            arr = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Array de imagem com dimensão inválida: shape={arr.shape}")
    arr = arr.astype("uint8")
    return Image.fromarray(arr).convert("RGB")

def encode_images(model, processor, dataset, image_column, device, batch_size=32):
    imagens_raw = dataset[image_column]
    pil_images = []
    for img in imagens_raw:
        try:
            pil = garantir_pil_rgb(img)
            pil_images.append(pil)
        except Exception:
            continue
    if not pil_images:
        return np.zeros((0, 1), dtype="float32")
    vecs = []
    for i in tqdm(range(0, len(pil_images), batch_size), desc="Embeddings de imagem"):
        chunk = pil_images[i:i + batch_size]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs)
        image_emb = image_emb.cpu().numpy()
        norms = np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-12
        image_emb = image_emb / norms
        vecs.append(image_emb.astype("float32"))
    if vecs:
        return np.concatenate(vecs, axis=0)
    return np.zeros((0, 1), dtype="float32")

def encode_texts(model, processor, textos, device, batch_size=64):
    vecs = []
    for i in tqdm(range(0, len(textos), batch_size), desc="Embeddings de texto"):
        chunk = textos[i:i + batch_size]
        inputs = processor(
            text=chunk,
            padding=True,
            truncation=True,
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
        text_emb = text_emb.cpu().numpy()
        norms = np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-12
        text_emb = text_emb / norms
        vecs.append(text_emb.astype("float32"))
    if vecs:
        return np.concatenate(vecs, axis=0)
    return np.zeros((0, 1), dtype="float32")

def compute_metrics(indices, gt, k_list=(1, 5, 10)):
    k_max = max(k_list)
    recalls = {k: [] for k in k_list}
    mrrs = []
    ndcgs = []
    for i, row in enumerate(indices):
        alvo = gt[i]
        hits = row[:k_max]
        rank = None
        for r, idx in enumerate(hits, start=1):
            if idx == alvo:
                rank = r
                break
        for k in k_list:
            if alvo in hits[:k]:
                recalls[k].append(1.0)
            else:
                recalls[k].append(0.0)
        if rank is not None:
            mrrs.append(1.0 / rank)
            if rank <= 10:
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                ndcgs.append(0.0)
        else:
            mrrs.append(0.0)
            ndcgs.append(0.0)
    out = {f"recall@{k}": float(np.mean(recalls[k])) for k in k_list}
    out["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0
    out["ndcg@10"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return out

def avaliar_modelo(model_id, dataset, text_column, image_column, device, max_eval_rows):
    if len(dataset) == 0:
        raise ValueError("Dataset vazio depois do carregamento.")
    n = min(max_eval_rows, len(dataset))
    dataset_eval = dataset.select(range(n))
    textos = dataset_eval[text_column]
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    image_embs = encode_images(model, processor, dataset_eval, image_column, device)
    text_embs = encode_texts(model, processor, textos, device)
    if image_embs.shape[0] == 0 or text_embs.shape[0] == 0:
        raise ValueError("Falha ao gerar embeddings.")
    if image_embs.shape[0] != text_embs.shape[0]:
        n_min = min(image_embs.shape[0], text_embs.shape[0])
        image_embs = image_embs[:n_min]
        text_embs = text_embs[:n_min]
    dim = image_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(image_embs.astype("float32"))
    k_max = 10
    _, I = index.search(text_embs.astype("float32"), k_max)
    gt = list(range(len(text_embs)))
    metrics = compute_metrics(I, gt, k_list=(1, 5, 10))
    return metrics

def descrever_nivel_performance(r10):
    if r10 < 0.1:
        return "muito baixo"
    if r10 < 0.3:
        return "baixo"
    if r10 < 0.6:
        return "moderado"
    return "alto"

def gerar_recomendacao(total_linhas, num_unicos_texto, df_resultados):
    partes = []
    partes.append(f"Tamanho total do dataset considerado: {total_linhas}")
    partes.append(f"Número de valores únicos na coluna de texto: {num_unicos_texto}")
    proporcao_unicos = num_unicos_texto / total_linhas if total_linhas > 0 else 0.0
    if total_linhas > 200 and (num_unicos_texto < 15 or proporcao_unicos < 0.1):
        partes.append("Há poucos valores distintos de texto em relação ao total de exemplos, o que se parece mais com um cenário de classificação.")
    else:
        partes.append("Os textos parecem relativamente diversos, o que é compatível com tarefas de retrieval ou geração.")
    df_ord = df_resultados.sort_values("recall@10", ascending=False)
    melhor = df_ord.iloc[0]
    melhor_modelo = str(melhor["modelo"])
    melhor_r10 = float(melhor["recall@10"])
    melhor_r1 = float(melhor["recall@1"])
    melhor_mrr = float(melhor["mrr"])
    nivel = descrever_nivel_performance(melhor_r10)
    partes.append(f"Melhor modelo nos testes: {melhor_modelo}")
    partes.append(f"Performance do melhor modelo: recall@1={melhor_r1:.4f}, recall@10={melhor_r10:.4f}, MRR={melhor_mrr:.4f} ({nivel}).")
    siglip_mask = df_resultados["modelo"].str.contains("siglip")
    clip_mask = df_resultados["modelo"].str.contains("clip")
    texto_familias = []
    if siglip_mask.any():
        r10_siglip = df_resultados.loc[siglip_mask, "recall@10"].mean()
        texto_familias.append(f"média recall@10 SigLIP={r10_siglip:.4f}")
    if clip_mask.any():
        r10_clip = df_resultados.loc[clip_mask, "recall@10"].mean()
        texto_familias.append(f"média recall@10 CLIP={r10_clip:.4f}")
    if texto_familias:
        partes.append("Comparando famílias de modelos: " + " | ".join(texto_familias))
        if siglip_mask.any() and clip_mask.any():
            r10_siglip = df_resultados.loc[siglip_mask, "recall@10"].mean()
            r10_clip = df_resultados.loc[clip_mask, "recall@10"].mean()
            if r10_siglip > r10_clip + 0.02:
                partes.append("Nos testes, os modelos SigLIP ficaram melhores em média que os CLIP para este dataset.")
            elif r10_clip > r10_siglip + 0.02:
                partes.append("Nos testes, os modelos CLIP ficaram melhores em média que os SigLIP para este dataset.")
            else:
                partes.append("Nos testes, SigLIP e CLIP tiveram desempenhos médios parecidos neste dataset.")
    if melhor_r10 < 0.3:
        if total_linhas < 300:
            partes.append("Os resultados de retrieval estão na faixa baixa e o dataset é pequeno (menos de 300 exemplos). A prioridade aqui pode ser aumentar o conjunto de dados antes de investir em fine-tuning.")
        elif total_linhas <= 100000:
            partes.append("Os resultados de retrieval estão na faixa baixa e o dataset tem tamanho razoável (acima de 300 exemplos). Vale considerar um fine-tuning de um dos modelos que se saiu melhor, focado neste domínio.")
        else:
            partes.append("Os resultados de retrieval estão na faixa baixa, mas o dataset é muito grande (mais de 100k exemplos). Pode ser interessante uma abordagem mais robusta, como pré-treinamento adicional, amostragem cuidadosa ou técnicas específicas para grandes volumes.")
    elif melhor_r10 < 0.6:
        partes.append("Os resultados de retrieval estão em um nível intermediário. Um fine-tuning direcionado provavelmente consegue melhorar a qualidade das recuperações.")
    else:
        partes.append("Os resultados de retrieval já estão em um nível alto. Um fine-tuning pode trazer ganhos menores e mais específicos, mas o modelo já é funcional para uso prático.")
    if total_linhas > 200 and (num_unicos_texto < 15 or proporcao_unicos < 0.1):
        partes.append("Como o padrão dos textos lembra uma tarefa de classificação, outra possibilidade é treinar um modelo de classificação usando essas classes como rótulos, em vez de focar apenas em retrieval.")
    return "\n".join(partes)

def pipeline(dataset_name, splits_str, text_column, image_column, max_rows):
    try:
        set_seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = carregar_dataset(dataset_name, splits_str)
        if text_column not in dataset.column_names:
            raise ValueError(f"Coluna de texto '{text_column}' não encontrada. Colunas disponíveis: {dataset.column_names}")
        if image_column not in dataset.column_names:
            raise ValueError(f"Coluna de imagem '{image_column}' não encontrada. Colunas disponíveis: {dataset.column_names}")
        total_linhas, num_unicos = analisar_dataset_base(dataset, text_column)
        if max_rows is None or max_rows <= 0:
            max_rows_uso = MAX_EVAL_ROWS
        else:
            max_rows_uso = min(int(max_rows), MAX_EVAL_ROWS)
        resultados = []
        for model_id in MODEL_IDS:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            metrics = avaliar_modelo(model_id, dataset, text_column, image_column, device, max_rows_uso)
            linha = {
                "modelo": model_id,
                "recall@1": metrics["recall@1"],
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "mrr": metrics["mrr"],
                "ndcg@10": metrics["ndcg@10"]
            }
            resultados.append(linha)
        if not resultados:
            raise ValueError("Nenhum resultado de modelo foi gerado.")
        df_resultados = pd.DataFrame(resultados)
        recomendacao = gerar_recomendacao(total_linhas, num_unicos, df_resultados)
        return recomendacao, df_resultados
    except Exception as e:
        return f"Ocorreu um erro durante a execução: {str(e)}", pd.DataFrame()

with gr.Blocks() as demo:
    gr.Markdown("# Demo de análise de dataset multimodal e avaliação de retrieval")
    with gr.Row():
        with gr.Column():
            dataset_name = gr.Textbox(
                label="Nome do dataset no Hugging Face",
                value="tungdop2/pokemon"
            )
            splits_str = gr.Textbox(
                label="Splits a usar (ex: train ou train,validation)",
                value="train"
            )
            text_column = gr.Textbox(
                label="Nome da coluna de texto",
                value="caption"
            )
            image_column = gr.Textbox(
                label="Nome da coluna de imagem",
                value="image"
            )
            max_rows = gr.Number(
                label="Máximo de linhas para avaliação (amostragem)",
                value=1000,
                precision=0
            )
            botao = gr.Button("Rodar análise")
        with gr.Column():
            saida_recomendacao = gr.Textbox(label="Resumo e recomendações", lines=14)
            saida_metricas = gr.Dataframe(label="Métricas por modelo")
    botao.click(
        fn=pipeline,
        inputs=[dataset_name, splits_str, text_column, image_column, max_rows],
        outputs=[saida_recomendacao, saida_metricas]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
