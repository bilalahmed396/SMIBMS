import gradio as gr
import torch
from transformers import RobertaTokenizer
from src.model import HybridBiLSTMRoBERTa, research_clean_text

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = HybridBiLSTMRoBERTa().to(device)
# Note: Ensure you have your trained 'model.pth' file locally to load weights
# model.load_state_dict(torch.load('model.pth', map_location=device))

def predict_sentiment(tweet):
    if not tweet.strip():
        return "Please enter a tweet!"
    
    clean_text = research_clean_text(tweet)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, 
                       max_length=128, padding='max_length').to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(inputs['input_ids'], inputs['attention_mask'])
        prediction = torch.argmax(output, dim=1).item()
    
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels.get(prediction, "Unknown")

# Launch Gradio
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter an airline tweet here..."),
    outputs="text",
    title="SMIBMS: Airline Sentiment Predictor",
    description="Enter a tweet to see if the sentiment is Positive, Negative, or Neutral."
)

if __name__ == "__main__":
    interface.launch()
