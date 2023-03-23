import torch
from transformers import BertTokenizer, BertForSequenceClassification
import subprocess


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Fine-tune the BERT model on a text classification task
train_data = [("open chrome", 1), ("launch firefox", 1), ("start notepad", 1), ("close the window", 0)]
train_texts = [text for text, label in train_data]
train_labels = [label for text, label in train_data]
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
train_labels = torch.tensor(train_labels)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**train_encodings, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Use the fine-tuned model to classify a user's command
text = "open the Chrome browser"
encoding = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
outputs = model(**encoding)
predicted_label = torch.argmax(outputs.logits, dim=1).item()
if predicted_label == 1:
    # launch the Chrome browser
    subprocess.call(["C:\Program Files\Google\Chrome\Application\chrome.exe"])
