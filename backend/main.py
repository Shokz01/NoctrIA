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
    {"role": "system", "content": (
        "Você é um professor especialista em criar quizzes educativos. "
        "Use exclusivamente as informações presentes no texto fornecido. "
        "Não invente fatos, nomes, definições, números ou explicações. "
        "Antes de criar cada pergunta, confirme que a resposta pode ser "
        "encontrada claramente no texto. Se não puder, escolha outro assunto "
        "que esteja explicitamente explicado no texto. "
        "Crie perguntas claras, relevantes e diferentes entre si. "
        "Cada pergunta deve ter exatamente quatro alternativas plausíveis, "
        "identificadas por A), B), C) e D), com apenas uma resposta correta. "
        "Todas as alternativas devem estar relacionadas ao assunto, mas "
        "somente uma deve responder corretamente à pergunta conforme o texto. "
        "Não crie alternativas absurdas ou com termos inventados. "
        "Use português brasileiro correto e natural. "
        "Não misture português com inglês sem necessidade. "
        "Distribua as letras das respostas corretas entre A, B, C e D "
        "sempre que possível. "
        "Responda exclusivamente com JSON válido, sem Markdown ou explicações extras."
    )},
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

    print("RESPOSTA BRUTA DO QUIZ:")
    print(resposta)

    print("Tokens gerados:", len(novos_tokens))
    print("Limite configurado:", 700)
    print("Fim da resposta:", repr(resposta[-80:]))

    if resposta.startswith('{"perguntas"') and resposta.endswith('}}'):
        resposta = resposta[:-1] + ']}'
    elif resposta.startswith('{"perguntas"') and resposta.endswith('}]'):
        resposta += '}'

    print(repr(resposta))
    print("Tamanho da resposta:", len(resposta))
    print("Caractere 698:", repr(resposta[698:710]))

    try:
        json.loads(resposta)
        print("JSON válido!")
    except json.JSONDecodeError as erro:
        print("Erro JSON:", erro)
        print("Trecho do erro:", repr(resposta[erro.pos - 40:erro.pos + 40]))

    print("Últimos códigos:", [ord(c) for c in resposta[-10:]])
    print("Último caractere:", repr(resposta[-1]))

    
    
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

