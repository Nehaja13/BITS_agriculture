@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    # Here you can add custom logic for different responses
    if "help" in user_message.lower():
        response = "To detect a crop disease, please upload a picture of the crop leaf."
    else:
        response = "I'm here to assist you with crop disease predictions. Try uploading a crop image!"
    return jsonify({"response": response})
