from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, RobertaForQuestionAnswering, AutoModelForTokenClassification, pipeline
import torch

app = Flask(__name__)

# Sample HTML template for file upload
upload_form = """
<!doctype html>
<title>Upload a File or Enter Text</title>
<h1>Upload a File or Enter Text</h1>
<form action="" method=post enctype=multipart/form-data>
  <input type=file name=file>
  <br><br>
  <textarea name=text rows="10" cols="30" placeholder="Enter your text here"></textarea>
  <br><br>
  <input type=submit value=Submit>
</form>
"""

# Route to handle file upload
@app.route("/", methods=["GET", "POST"])
def upload_or_enter_text():
    if request.method == "POST":
        file = request.files["file"]
        text = request.form.get("text")

        if file and file.filename:
            content = file.read().decode("utf-8")
        elif text:
            content = text
        else:
            content = ""

        texts = process_content(text, content)
        return render_template_string("""
            <h1>Processed Content</h1>
            <p>Summary: {{ text1 }}</p>
            <p>Entities: {{ text2 }}</p>
            <p>Query Answer: {{ text3 }}</p>
            <a href="/">Upload another file or enter more text</a>
        """, text1=texts[0], text2=texts[1], text3=texts[2])
    
    return render_template_string(upload_form)

def load_models():
    try:
        print("Loading Summarizer...")
        summarizer_tokenizer = AutoTokenizer.from_pretrained("pszemraj/pegasus-x-large-book-summary")
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/pegasus-x-large-book-summary")
        print("Loaded Summarizer")

        print("Loading Question-Answering...")
        qa_tokenizer = AutoTokenizer.from_pretrained("gustavhartz/roberta-base-cuad-finetuned")
        qa_model = RobertaForQuestionAnswering.from_pretrained("gustavhartz/roberta-base-cuad-finetuned")
        print("Loaded Question-Answering")

        print("Loading NER...")
        ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
        print("Loaded NER")

        return summarizer_tokenizer, summarizer_model, qa_tokenizer, qa_model, ner_tokenizer, ner_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None, None
    
summarizer_tokenizer, summarizer_model, qa_tokenizer, qa_model, ner_tokenizer, ner_model = load_models()
    
def summarize_text(document):
    try:
        inputs = summarizer_tokenizer(document, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summarizer_model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error in summarize_text: {e}")
        return "Error in summarizing text"

def extract_entities(text):
    try:
        process = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
        entities = process(text)
        end_text = []
        for i in entities:
            addition = i['word'] + ': ' + i['entity'][2:]
            end_text.append(addition)

        return '\n'.join(end_text)
    except Exception as e:
        print(f"Error in extract_entities: {e}")
        return ["Error in extracting entities"]

def answer_question(question, text):
    try:
        inputs = qa_tokenizer(question, text, return_tensors="pt", max_length=512, truncation=True)
        outputs = qa_model(**inputs)
        start_scores, end_scores = outputs.start_logits, outputs.end_logits
        all_tokens = qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])
        return answer
    except Exception as e:
        print(f"Error in answer_question: {e}")
        return "Error in answering question"

def process_content(question, content):
    # Process the file content and extract three text segments
    #lines = content.splitlines()
    text1 = summarize_text(content)
    text2 = ' '.join(extract_entities(content))
    text3 = answer_question(question, content)
    return text1, text2, text3

if __name__ == "__main__":
    app.run(debug=True)