
const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatContainer = document.getElementById("chat-container");

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

function addMessage(text, sender) {
    const div = document.createElement("div");
    div.classList.add("message", sender);
    div.textContent = text;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const pergunta = userInput.value.trim();
    if (!pergunta) return;

    addMessage(pergunta, "user");
    userInput.value = "";

    try {
        const response = await fetch("http://127.0.0.1:8000/responder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pergunta })
        });

        if (!response.ok) throw new Error("Erro na requisição");

        const data = await response.json();
        addMessage(data.resposta, "bot");
    } catch (error) {
        addMessage("Desculpe, não consegui processar sua pergunta. Verifique se o servidor está rodando.", "bot");
    }
}
