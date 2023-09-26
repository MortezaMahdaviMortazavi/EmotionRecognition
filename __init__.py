def predict(texts, local_ids, model, device, threshold=0.5):
    model.eval()  # Set the model to evaluation 

    inputs = []  # To store the input tensors
    label_dict = {0: 'sadness', 1: 'happiness', 2: 'surprise', 3: 'fear', 4: 'disgust', 5: 'anger', 6: 'other'}
    num_classes = len(label_dict)

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    max_length = 64
    for text in tqdm(texts):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze().to(device)
        attention_mask = encoding['attention_mask'].squeeze().to(device)
        inputs.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    with torch.no_grad():
        predictions = []
        prob_matrix = []  # To store the probability distribution for each text
        for input_data in tqdm(inputs):
            input_ids = input_data['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
            attention_mask = input_data['attention_mask'].unsqueeze(0).to(device)  # Add batch dimension

            outputs = model(input_ids, attention_mask)
            predicted_probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            prob_matrix.append(predicted_probs)

            # Determine primary emotion based on probability threshold
            max_prob = torch.max(outputs, dim=1)[0].item()
            if max_prob > threshold:
                predicted_label = label_dict[torch.argmax(outputs, dim=1).item()]
            else:
                predicted_label = 'other'
            predictions.append(predicted_label)

    csv_data = []
    for local_id, text, primary_emotion, probs in zip(local_ids, texts, predictions, prob_matrix):
        emotion_probs = [1 if prob > threshold else 0 for prob in probs]
        row = [local_id, text, primary_emotion] + emotion_probs
        csv_data.append(row)

    # Create a DataFrame from the CSV data
    columns = ["local_id", "tweet", "primary_emotion", "anger", "sadness", "fear", "happiness", "disgust", "surprise", "other"]
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv('predictions_large_gru.csv', index=False)

    return predictions
