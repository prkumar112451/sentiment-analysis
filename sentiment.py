import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import gc

# 1. Setup device: use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 2. Huggingface login (set your HF token as an environment variable for safety)
hf_access_token = os.environ.get('HF_TOKEN', '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')  # Replace or set env variable
login(token=hf_access_token)

# 3. Model + label meta-data
language_model_dict = {
    "en": "cardiffnlp/twitter-roberta-base-sentiment",
}
label_mapping_dict = {
    "en": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
}

# 4. Load models/tokenizers (on correct device!)
model_tokenizer_dict = {}
for lang_code, model_name in language_model_dict.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()
    model_tokenizer_dict[lang_code] = {"model": model, "tokenizer": tokenizer}


def create_chunks(sentences, max_words=5000):
    """
    sentences: list of (id, text) tuples.
    Returns list of (id, text) tuples, batched not exceeding max_words.
    """
    output = []
    current_chunk = []
    current_length = 0
    for idx, (serial, s) in enumerate(sentences):
        sentence_length = len(s.split())
        if current_length + sentence_length > max_words and current_chunk:
            output.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append((serial, s))
        current_length += sentence_length
    if current_chunk:
        output.append(current_chunk)
    return output


def sentiment_final(chunks, language_code):
    """
    Takes a batch of (id, text) and returns sentiment scores for each.
    """
    selected_model_info = model_tokenizer_dict[language_code]
    tokenizer = selected_model_info["tokenizer"]
    model = selected_model_info["model"]

    sentences = [sentence for _, sentence in chunks]
    encoding = tokenizer(
        sentences, max_length=512, truncation=True, padding=True, return_tensors='pt'
    )
    # Move input tensors to the same device as model
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        scores = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

    label_mapping = label_mapping_dict.get(language_code, {})
    result = []
    for i, (serial, _) in enumerate(chunks):
        chunk_scores = scores[i]
        label_scores = []
        for j, score in enumerate(chunk_scores):
            label = model.config.id2label[j] if hasattr(model.config, "id2label") else f"LABEL_{j}"
            human_label = label_mapping.get(label, label)
            label_scores.append({'label': human_label, 'score': score, 'id': serial})
        result.append(label_scores)
    return result


def extract_id_and_texts(texts_data):
    """
    Accepts a list of dicts with keys {'id', 'document'}, or just list of strings,
    and returns a list of (id, text) tuples.
    """
    output = []
    if not texts_data:
        return output
    if isinstance(texts_data[0], dict):
        for item in texts_data:
            if 'id' not in item or 'document' not in item:
                raise ValueError("Dictionary input must have 'id' and 'document' keys")
            output.append((item['id'], item['document']))
    elif isinstance(texts_data[0], str):
        for idx, item in enumerate(texts_data):
            output.append((idx, item))
    else:
        raise ValueError("Unsupported input format for texts_data")
    return output


def sentiment_text(texts_data, language_code):
    if language_code not in model_tokenizer_dict:
        return [{"error": f"Model for language code '{language_code}' not found"}]

    result = []
    try:
        id_text_tuples = extract_id_and_texts(texts_data)
        chunks_list = create_chunks(id_text_tuples)
        for chunks in chunks_list:
            try:
                sentiments = sentiment_final(chunks, language_code)
                for idx, sentiment_scores in enumerate(sentiments):
                    serial = chunks[idx][0]
                    text = chunks[idx][1]
                    sorted_scores = sorted(sentiment_scores, key=lambda x: x['label'])
                    result.append({
                        "id": serial,
                        "text": text,
                        "sentiment": sorted_scores
                    })
            except Exception as e:
                result.append({"error": str(e)})
        result.sort(key=lambda x: x['id'])
        return result
    finally:
        # Optional: clean only if you're sure memory usage is high
        if 'result' in locals():
            del result
        gc.collect()
        torch.cuda.empty_cache()
