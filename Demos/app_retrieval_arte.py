import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
import gradio as gr

MODEL_NAME = "turing552/clip-wikiart-raw-v1-10ep"
DATASET_NAME = "Artificio/WikiArt"
DATASET_SPLIT = "train"
TEXT_COLUMN_ORIGINAL = "description"
IMAGE_COLUMN = "image"

def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    else:
        return torch.device("cpu"), torch.float32

def load_wikiart_dataset():
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    images = [example[IMAGE_COLUMN].convert("RGB") for example in dataset]
    return dataset, images

def encode_images_clip(images, model, processor, device, dtype, batch_size=32):
    all_embeds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
            image_embeds = model.get_image_features(pixel_values=pixel_values)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(image_embeds.cpu())
    return torch.cat(all_embeds, dim=0)

def encode_text_clip(text, model, processor, device):
    model.eval()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_embeds = model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds.cpu()[0]

def retrieve_top_k_images(query, image_embeds, images, dataset, model, processor, device, k=10):
    query = query.strip()
    if not query:
        return [], "Digite uma descrição para realizar a busca."
    text_emb = encode_text_clip(query, model, processor, device)
    similarities = torch.matmul(image_embeds, text_emb)
    topk_values, topk_indices = torch.topk(similarities, k=min(k, len(images)))
    retrieved_images = []
    captions_info = []
    for rank, (score, idx) in enumerate(zip(topk_values.tolist(), topk_indices.tolist()), start=1):
        retrieved_images.append(images[idx])
        example = dataset[idx]
        title = example.get("title", "")
        artist = example.get("artist", "")
        date = example.get("date", "")
        genre = example.get("genre", "")
        style = example.get("style", "")
        description = example.get(TEXT_COLUMN_ORIGINAL, "")
        info = (
            f"Rank {rank} | Similaridade: {score:.4f}\n"
            f"Título: {title}\n"
            f"Artista: {artist}\n"
            f"Data: {date}\n"
            f"Gênero: {genre}\n"
            f"Estilo: {style}\n"
            f"Descrição: {description}"
        )
        captions_info.append(info)
    info_text = "\n\n----------------------------------------\n\n".join(captions_info)
    return retrieved_images, info_text

device, dtype = get_device_and_dtype()

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)

dataset_wikiart, wikiart_images = load_wikiart_dataset()
wikiart_image_embeds = encode_images_clip(
    wikiart_images,
    model,
    processor,
    device,
    dtype,
    batch_size=32
)

def wikiart_retrieval_interface(user_query):
    images, info = retrieve_top_k_images(
        user_query,
        wikiart_image_embeds,
        wikiart_images,
        dataset_wikiart,
        model,
        processor,
        device,
        k=10
    )
    return images, info

iface = gr.Interface(
    fn=wikiart_retrieval_interface,
    inputs=gr.Textbox(lines=3, label="Descreva uma obra, estilo ou cena"),
    outputs=[
        gr.Gallery(label="Top 10 imagens recuperadas", columns=5, height=500),
        gr.Textbox(label="Informações das imagens recuperadas")
    ],
    title="Retrieval Multimodal WikiArt",
    description="Digite um texto e o sistema recupera as 10 imagens mais similares do dataset Artificio/WikiArt usando CLIP fine-tunado."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
