import requests

url = "http://127.0.0.1:5000/chatbot"
data = {"hotel": "A", "message": "WiFi"}
response = requests.post(url, json=data)

print(response.json())  # Should print chatbot's response
