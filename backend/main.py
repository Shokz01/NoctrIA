import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class PerguntaQuiz(BaseModel):
    pergunta: str
    alternativas: list[str]
    resposta_correta: str

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)

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
quiz_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Carregando tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(token_name)

print("Carregando modelo...")
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("Modelo carregado com sucesso!")

print("Carregando modelo de quiz...")

quiz_tokenizer = AutoTokenizer.from_pretrained(quiz_model_name)

quiz_model = AutoModelForCausalLM.from_pretrained(quiz_model_name)

print("Modelo de quiz carregado com sucesso!")

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



@app.post("/quiz")
def gerar_quiz(dados: TextoEntrada):
    texto = dados.texto.strip()

    if not texto:
        return {"erro": "Digite um texto para gerar o quiz."}

    mensagens = [
    {
    "role": "system",
    "content": """
Você é um professor especialista em criar questões educacionais.

Sua tarefa é criar exatamente 5 perguntas de múltipla escolha com base EXCLUSIVAMENTE no texto fornecido pelo usuário.

REGRAS OBRIGATÓRIAS:

1. Todas as perguntas devem ser respondíveis usando informações presentes no texto.
2. Nunca invente informações, dados, explicações ou acontecimentos.
3. Não faça perguntas sobre informações que não aparecem no texto.
4. Cada pergunta deve abordar um aspecto diferente do conteúdo.
5. Cada pergunta deve possuir exatamente 4 alternativas.
6. As alternativas devem ser identificadas como A), B), C) e D).
7. Apenas uma alternativa pode estar correta.
8. As alternativas incorretas devem ser plausíveis e relacionadas ao assunto.
9. Nunca repita alternativas.
10. Não use palavras ou conceitos aleatórios que não tenham relação com o texto.
11. A resposta_correta deve conter somente a letra da alternativa correta: A, B, C ou D.
12. Revise todas as perguntas e respostas antes de finalizar.

FORMATO DE SAÍDA:

Retorne exclusivamente um JSON válido, sem Markdown, sem explicações e sem texto adicional.

Use exatamente esta estrutura:

{
    "perguntas": [
        {
            "pergunta": "Texto da pergunta",
            "alternativas": [
                "A) Alternativa",
                "B) Alternativa",
                "C) Alternativa",
                "D) Alternativa"
            ],
            "resposta_correta": "A"
        }
    ]
}
"""
},
        {
            "role": "user",
            "content": (
                "Crie exatamente 5 perguntas de múltipla escolha com base no texto abaixo. "
                "Cada pergunta deve ter 4 alternativas identificadas por A), B), C) e D). "
                "O campo resposta_correta deve conter somente a letra da alternativa correta. "
                "Distribua as letras corretas entre as perguntas sempre que possível. "
                'Use esta estrutura JSON: '
                '{"perguntas":[{"pergunta":"...","alternativas":["A) ...","B) ...",'
                '"C) ...","D) ..."],"resposta_correta":"A"}]}\n\n'
                f"Texto:\n{texto}"
            )
        }
    ]

    prompt = quiz_tokenizer.apply_chat_template(
        mensagens,
        tokenize=False,
        add_generation_prompt=True
    )

    entradas = quiz_tokenizer(prompt, return_tensors="pt")

    saida = quiz_model.generate(
        **entradas,
        max_new_tokens=700,
        do_sample=False,
        pad_token_id=quiz_tokenizer.eos_token_id
    )

    novos_tokens = saida[0][entradas["input_ids"].shape[1]:]

    resposta = quiz_tokenizer.decode(
        novos_tokens,
        skip_special_tokens=True
    ).strip()

    # Remove blocos Markdown que o modelo pode adicionar
    resposta = resposta.strip()

    if resposta.startswith("```"):
        linhas = resposta.splitlines()

        if linhas[0].startswith("```"):
            linhas = linhas[1:]

        if linhas and linhas[-1].strip() == "```":
            linhas = linhas[:-1]

        resposta = "\n".join(linhas).strip()

    print("RESPOSTA BRUTA DO QUIZ:")
    print(resposta)

    try:
        quiz = json.loads(resposta)

        if isinstance(quiz, str):
            quiz = json.loads(quiz)

        if not isinstance(quiz, dict) or "perguntas" not in quiz:
            raise ValueError("O JSON não contém a lista de perguntas.")

        return quiz


    except (json.JSONDecodeError, ValueError) as erro:
        return {
            "erro": str(erro),
            "resposta_modelo": resposta
        }

