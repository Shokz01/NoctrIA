
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://127.0.0.1:3000",
    "http://localhost:3000",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

token_name = "unicamp-dl/ptt5-base-portuguese-vocab"
model_name = "recogna-nlp/ptt5-base-summ"

print("Carregando tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(token_name)

print("Carregando modelo...")
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("Modelo carregado com sucesso!")


class TextoEntrada(BaseModel):
    texto: str


@app.post("/resumir")
def resumir(dados: TextoEntrada):
    texto = dados.texto.strip()

    if not texto:
        return {"erro": "Digite um texto para resumir."}

    inputs = tokenizer.encode(
        texto,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs,
        max_length=256,
        min_length=32,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    resumo = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return {"resumo": resumo}
