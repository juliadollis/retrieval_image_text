import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel, LlavaForConditionalGeneration, LlavaProcessor, CLIPImageProcessor, LlamaTokenizer
import gradio as gr

def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    else:
        return torch.device("cpu"), torch.float32

def load_pokemon_dataset():
    dataset = load_dataset("tungdop2/pokemon", split="train")
    images = [example["image"].convert("RGB") for example in dataset]
    return dataset, images

def encode_images_siglip(images, model, processor, device, dtype, batch_size=32):
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

def encode_text_siglip(text, model, processor, device):
    model.eval()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_embeds = model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds.cpu()[0]

def retrieve_pokemon(query, image_embeds, dataset, images, model, processor, device):
    text_emb = encode_text_siglip(query, model, processor, device)
    similarities = torch.matmul(image_embeds, text_emb)
    best_score, best_idx = torch.max(similarities, dim=0)
    best_idx = best_idx.item()
    best_score = best_score.item()
    example = dataset[best_idx]
    image = images[best_idx]
    name = example.get("name", "")
    type_1 = example.get("type_1", "")
    type_2 = example.get("type_2", None)
    caption = example.get("caption", "")
    return best_idx, best_score, image, name, type_1, type_2, caption

def generate_llava_answer(llava_model, llava_processor, image, user_query, name, type_1, type_2, caption, dtype):
    if type_2 is None:
        type_2_text = "nenhum"
    else:
        type_2_text = str(type_2)
    prompt_text = (
        "Você é um assistente do tipo Pokédex.\n"
        f"Descrição fornecida pelo usuário: {user_query}\n"
        f"Nome do Pokémon no card: {name}.\n"
        f"Tipo 1: {type_1}.\n"
        f"Tipo 2: {type_2_text}.\n"
        f"Descrição visual da carta: {caption}.\n"
        "Explique em poucas frases, em português, que tipo de Pokémon é esse, "
        "mencione claramente os tipos e faça uma descrição amigável baseada apenas nessas informações."
    )
    formatted_prompt = "USER: <image>\n" + prompt_text + " ASSISTANT:"
    inputs = llava_processor(images=image, text=formatted_prompt, return_tensors="pt", padding=True)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                inputs[k] = v.to(llava_model.device, dtype=dtype)
            else:
                inputs[k] = v.to(llava_model.device)
    with torch.no_grad():
        generate_ids = llava_model.generate(**inputs, max_new_tokens=128)
    decoded = llava_processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return decoded

device, dtype = get_device_and_dtype()
siglip_model_id = "google/siglip-so400m-patch14-384"
llava_model_id = "llava-hf/llava-1.5-7b-hf"

siglip_processor = AutoProcessor.from_pretrained(siglip_model_id)
siglip_model = AutoModel.from_pretrained(siglip_model_id, torch_dtype=dtype).to(device)

llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_model_id,
    torch_dtype=dtype,
    device_map="auto"
)

llava_image_processor = CLIPImageProcessor.from_pretrained(llava_model_id)
llava_tokenizer = LlamaTokenizer.from_pretrained(llava_model_id)
llava_processor = LlavaProcessor(image_processor=llava_image_processor, tokenizer=llava_tokenizer)

dataset_pokemon, pokemon_images = load_pokemon_dataset()
pokemon_image_embeds = encode_images_siglip(
    pokemon_images,
    siglip_model,
    siglip_processor,
    device,
    dtype,
    batch_size=32
)

def rag_pokemon_interface(user_query):
    user_query = user_query.strip()
    if not user_query:
        return None, "Digite uma descrição do Pokémon para começar.", ""
    idx, score, image, name, type_1, type_2, caption = retrieve_pokemon(
        user_query,
        pokemon_image_embeds,
        dataset_pokemon,
        pokemon_images,
        siglip_model,
        siglip_processor,
        device
    )
    if type_2 is None:
        type_2_print = "nenhum"
    else:
        type_2_print = str(type_2)
    info_md = (
        f"Nome: {name}\n\n"
        f"Tipo 1: {type_1}\n\n"
        f"Tipo 2: {type_2_print}\n\n"
        f"Score de similaridade (cos): {score:.4f}\n\n"
        f"Caption do dataset:\n{caption}"
    )
    llava_answer = generate_llava_answer(
        llava_model,
        llava_processor,
        image,
        user_query,
        name,
        type_1,
        type_2,
        caption,
        dtype
    )
    return image, info_md, llava_answer

iface = gr.Interface(
    fn=rag_pokemon_interface,
    inputs=gr.Textbox(lines=3, label="Descreva o Pokémon"),
    outputs=[
        gr.Image(type="pil", label="Imagem recuperada"),
        gr.Markdown(label="Informações do Pokémon"),
        gr.Markdown(label="Resposta da LLaVA")
    ],
    title="RAG Pokémon com SigLIP + LLaVA",
    description="Digite uma descrição do Pokémon e o sistema recupera a carta mais parecida e explica em português."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=786, share=True)
