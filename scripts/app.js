const modeLinks = document.querySelectorAll("[data-mode]");
const form = document.getElementById("form");
const textInput = document.getElementById("itext");
const submitButton = document.getElementById("ibutton");
const result = document.getElementById("result");

let currentMode = "resumo";

const resumoLink = document.querySelector('[data-mode="resumo"]');

if (resumoLink) {
    resumoLink.classList.add("active");
    resumoLink.setAttribute("aria-current", "page");
}

const modes = {
    resumo: {
        title: "Criar resumo",
        placeholder: "Cole aqui o conteúdo que deseja resumir...",
        button: "Criar resumo"
    },

    quiz: {
        title: "Gerar quiz",
        placeholder: "Cole aqui o conteúdo para gerar perguntas...",
        button: "Gerar quiz"
    }
};


modeLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
        event.preventDefault();

        currentMode = link.dataset.mode;

        textInput.placeholder = modes[currentMode].placeholder;
        submitButton.textContent = modes[currentMode].button;

        modeLinks.forEach((item) => {
            item.classList.remove("active");
            item.removeAttribute("aria-current");
        });

        link.classList.add("active");
        link.setAttribute("aria-current", "page");

        result.textContent = "";
        textInput.focus();
    });
});


form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const texto = textInput.value.trim();


    if (!texto) {
        result.textContent = "Insira o conteúdo da sua matéria.";
        textInput.focus();
        return;
    }

    if (currentMode === "resumo") {
    result.textContent = "Gerando resumo...";

    try {
        const response = await fetch("http://127.0.0.1:8000/resumir", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ texto })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Erro ao gerar resumo.");
        }

        if (data.erro) {
            throw new Error(data.erro);
        }

        result.textContent = data.resumo;
    } catch (error) {
        result.textContent = `Erro: ${error.message}`;
    }

} else {
    result.textContent = "Gerando quiz...";

    try {
        const response = await fetch("http://127.0.0.1:8000/quiz", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ texto })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Erro ao gerar quiz.");
        }

        if (data.erro) {
            throw new Error(data.erro);
        }

        if (!Array.isArray(data.perguntas)) {
            throw new Error("A resposta da API não contém uma lista de perguntas.");
        }

        result.replaceChildren();

        data.perguntas.forEach((pergunta, index) => {
            const bloco = document.createElement("div");
            const titulo = document.createElement("h3");
            const alternativas = document.createElement("ul");
            const gabarito = document.createElement("p");

            titulo.textContent = `${index + 1}. ${pergunta.pergunta}`;

            pergunta.alternativas.forEach((alternativa) => {
                const item = document.createElement("li");
                item.textContent = alternativa;
                alternativas.appendChild(item);
            });

            gabarito.textContent =
                `Resposta esperada: ${pergunta.resposta_correta}`;

            bloco.append(titulo, alternativas, gabarito);
            result.appendChild(bloco);
        });

    } catch (error) {
        result.textContent = `Erro: ${error.message}`;
    }
}

});