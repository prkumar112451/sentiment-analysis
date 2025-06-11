import requests
import time
import json
from sentiment import sentiment_text

API_URL = "http://localhost:5001/api/SentimentRequest"
POLL_INTERVAL = 1  # seconds

def poll_for_request():
    resp = requests.get(API_URL)
    if resp.status_code != 200:
        print("No new requests or error:", resp.status_code, resp.text)
        return None
    return resp.json()

def send_result(result):
    print(result)
    resp = requests.post(API_URL, json=result)
    if 200 <= resp.status_code < 300:
        print("Result sent successfully")
    else:
        print("Failed to send result: ", resp.status_code, resp.text)

def convert_to_csharp_sentiment_model(py_results):
    out = []
    for item in py_results:
        # Get sentiment scores as a dict
        scores = {entry['label']: entry['score'] for entry in item['sentiment']}
        # Fill missing scores just in case
        confidence_positive = scores.get('positive', 0.0)
        confidence_neutral = scores.get('neutral', 0.0)
        confidence_negative = scores.get('negative', 0.0)
        # Which is the max?
        top_sentiment = max(scores, key=scores.get)
        out.append({
            'id': item['id'],
            'document': item['text'],  # or .get("document") if that's incoming
            'sentiment': top_sentiment,
            'confidence_scores_positive': confidence_positive,
            'confidence_scores_neutral': confidence_neutral,
            'confidence_scores_negative': confidence_negative
        })
    return out

def main():
    while True:
        try:
            req_body = poll_for_request()
            print(req_body)
            if not req_body or req_body.get("location") is None:
                time.sleep(POLL_INTERVAL)
                continue
            
            request_id = req_body["requestId"]
            file_path = req_body["location"]
            language_code = req_body.get("languageCode", "en")
            operation = req_body.get("operation", "sentiment")

            # Updated: Read and parse structured JSON with double layer
            with open(file_path, "r", encoding='utf-8') as f:
                file_content = f.read()
                root_obj = json.loads(file_content)
                payload_obj = json.loads(root_obj['Payload'])
                texts_data = payload_obj['sentiments']
            
            sentiment_results = sentiment_text(texts_data, language_code)
            sentiment_models = convert_to_csharp_sentiment_model(sentiment_results)

            response_payload = {
                "result": {
                    "code": 200,
                    "message": "completed",
                    "error": "",
                    "data": sentiment_models
                },
                "requestID": request_id,
                "status": "completed",
                "operation": operation
            }

            send_result(response_payload)

            send_result(response_payload)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    main()