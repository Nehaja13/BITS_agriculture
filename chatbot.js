// chat.js

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const imageInput = document.getElementById("image-input");

function addMessageToChatBox(sender, message) {
  const messageElement = document.createElement("div");
  messageElement.className = `message ${sender}`;
  messageElement.innerHTML = message;
  chatBox.appendChild(messageElement);
  chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
}

function sendMessage() {
  const message = userInput.value;
  if (message.trim()) {
    addMessageToChatBox("user", message);
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    })
      .then(response => response.json())
      .then(data => {
        addMessageToChatBox("bot", data.response);
      });
    userInput.value = ""; // Clear the input field
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    const formData = new FormData();
    formData.append("file", file);

    addMessageToChatBox("user", "Image uploaded. Analyzing...");
    
    fetch("/api/disease-predict", {
      method: "POST",
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        addMessageToChatBox("bot", `Disease Prediction: ${data.prediction}`);
      })
      .catch(error => {
        addMessageToChatBox("bot", "Error: Could not process image.");
      });
  }
});
