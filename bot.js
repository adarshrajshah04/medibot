async function sendMessage() {
  const input = document.getElementById("userInput");
  const message = input.value.trim();
  if (message === "") return;

  const messagesDiv = document.getElementById("messages");

  // Add user message
  const userMsg = document.createElement("div");
  userMsg.className = "message user";
  userMsg.textContent = message;
  messagesDiv.appendChild(userMsg);

  input.value = "";
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  // Loading indicator
  const botTyping = document.createElement("div");
  botTyping.className = "message bot";
  botTyping.textContent = "Typing...";
  messagesDiv.appendChild(botTyping);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  try {
    const response = await fetch("/get", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message })
    });

    const data = await response.json();
    botTyping.remove();

    const botReply = document.createElement("div");
    botReply.className = "message bot";
    botReply.textContent = data.reply;
    messagesDiv.appendChild(botReply);

  } catch (err) {
    botTyping.textContent = "⚠️ Error: Could not connect to server";
  }

  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
