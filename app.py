from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the summarization model and tokenizer
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        # Get the text from the POST request
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=False, max_length=None)
        input_ids = inputs["input_ids"][0]  # Extract token IDs

        # Split tokens into chunks
        max_input_tokens = 1024  # Token limit for the model
        chunks = [input_ids[i: i + max_input_tokens] for i in range(0, len(input_ids), max_input_tokens)]

        # Summarize each chunk and handle empty or invalid chunks
        summaries = []
        for chunk in chunks:
            if len(chunk) == 0:  # Skip empty chunks
                continue
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

            # Dynamically adjust max_length based on chunk size
            chunk_max_length = min(len(chunk_text.split()), 150)  # Set max_length to chunk's word count or 150

            try:
                summary = summarizer(chunk_text, max_length=chunk_max_length, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summaries.append("")

        # Combine summaries into a single result
        final_summary = " ".join(summaries)

        return jsonify({"summary": final_summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
