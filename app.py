import requests
import time
import json
from sentiment import sentiment_text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
API_URL = "http://localhost:5001/api/SentimentRequest"
POLL_INTERVAL = 1  # seconds

private_ip = None
computer_name = None

def get_network_info():
    global private_ip, computer_name
    if private_ip is None or computer_name is None:
        try:
            url = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
            headers = {"Metadata": "true"}            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                metadata = response.json()
                computer_name = metadata.get("compute", {}).get("name", "")
                private_ip = metadata.get("network", {}).get("interface", [])[0].get("ipv4", {}).get("ipAddress", [])[0].get("privateIpAddress", "")
        except Exception as e:
            logger.error(f"Error getting private IP address or computer name: {e}")
            private_ip = 'Unknown'
            computer_name = 'Unknown'
            
def poll_for_request():
    try:
        resp = requests.get(API_URL)
        if resp.status_code != 200:
            logger.info("No new requests or error: %s %s", resp.status_code, resp.text)
            return None
        return resp.json()
    except Exception as e:
        logger.error(f"Error polling for request: {e}")
        return None

def send_result(result):
    logger.info(result)
    try:
        resp = requests.post(API_URL, json=result)
        if 200 <= resp.status_code < 300:
            logger.info("Result sent successfully")
        else:
            logger.info("Failed to send result: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error(f"Error sending result: {e}")

def convert_to_csharp_sentiment_model(py_results):
    out = []
    for item in py_results:
        if "sentiment" not in item:  # in case of error object
            continue
        scores = {entry['label']: entry['score'] for entry in item['sentiment']}
        confidence_positive = scores.get('positive', 0.0)
        confidence_neutral = scores.get('neutral', 0.0)
        confidence_negative = scores.get('negative', 0.0)
        top_sentiment = max(scores, key=scores.get, default="neutral")
        out.append({
            'id': item['id'],
            'document': item['text'],
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
            logger.info(req_body)
            if not req_body or req_body.get("location") is None:
                time.sleep(POLL_INTERVAL)
                continue
            
            request_id = req_body["requestId"]
            file_path = req_body["location"]
            language_code = req_body.get("languageCode", "en")
            operation = req_body.get("operation", "sentiment")

            # Read and parse structured JSON
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

        except Exception as e:
            logger.error(f"Error: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    get_network_info()
    if private_ip and private_ip != 'Unknown':
        API_URL = f"http://{private_ip}:5001/api/SentimentRequest"
    else:
        logger.warning("Using fallback API_URL: %s", API_URL)
    main()
