import requests

url = "https://uptor-ds-ml.onrender.com/predict"

data = {
    "size": 1700
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.text)