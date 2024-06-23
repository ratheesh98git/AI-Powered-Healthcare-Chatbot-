import nltk
from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
nltk.download('punkt')

app = Flask(__name__)

# Load pre-trained model and tokenizer for the QA pipeline
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def get_healthcare_context():
    context = """
    Welcome to the AI-Powered Healthcare Chatbot! Here, you can ask about symptoms, medications, and get general health advice.

    Symptom Checker:
    - Headache: A headache could be caused by stress, dehydration, or more serious conditions such as migraines or infections.
    - Fever: A fever is often a sign of infection. It's your body's way of fighting off illness.
    - Cough: Coughing can be a symptom of a common cold, flu, or more serious conditions like bronchitis or pneumonia.

    Medication Information:
    - Ibuprofen: Used for pain relief, reducing fever, and anti-inflammation. Possible side effects include stomach pain and dizziness.
    - Paracetamol: Commonly used for pain relief and reducing fever. Overdose can cause serious liver damage.

    Health Tips:
    - Drink plenty of water every day.
    - Maintain a balanced diet with plenty of fruits and vegetables.
    - Exercise regularly to maintain physical and mental health.

    Appointment Scheduling:
    - To schedule an appointment, please provide your preferred date and time, and we will assist you.

    Please ask your question!
    """
    return context

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Healthcare Chatbot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                background-color: #f5f5f5;
            }
            #chat-container {
                width: 50%;
                height: 60%;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
            }
            #chat-window {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                border-bottom: 1px solid #eee;
            }
            #user-input {
                display: flex;
            }
            #user-input input {
                flex: 1;
                padding: 10px;
                border: none;
                border-top: 1px solid #eee;
                border-bottom-left-radius: 10px;
            }
            #user-input button {
                padding: 10px;
                border: none;
                background-color: #007bff;
                color: white;
                cursor: pointer;
                border-bottom-right-radius: 10px;
            }
            #chat-window div {
                margin-bottom: 10px;
            }
            .user-message {
                text-align: right;
            }
            .bot-message {
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div id="chat-window"></div>
            <div id="user-input">
                <input type="text" id="message" placeholder="Type your message here...">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        <script>
            function sendMessage() {
                var message = document.getElementById("message").value;
                if (message.trim() === "") return;
                
                var userMessageDiv = document.createElement("div");
                userMessageDiv.className = "user-message";
                userMessageDiv.textContent = message;
                document.getElementById("chat-window").appendChild(userMessageDiv);
                
                fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    var botMessageDiv = document.createElement("div");
                    botMessageDiv.className = "bot-message";
                    botMessageDiv.textContent = data.response;
                    document.getElementById("chat-window").appendChild(botMessageDiv);
                    document.getElementById("message").value = "";
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    context = get_healthcare_context()
    result = qa_pipeline(question=user_input, context=context)
    answer = result['answer']
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)
