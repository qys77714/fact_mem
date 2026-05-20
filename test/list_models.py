import http.client
import json
from pathlib import Path

conn = http.client.HTTPSConnection("api.ai-gaochao.cn")
payload = ''
headers = {
   'Authorization': 'Bearer sk-yx5VeS4xYwchVVMO1e6fEdEcCfF448B68e65Bb31E9F6Ba97'
}
conn.request("GET", "/v1/models", body=payload, headers=headers)
res = conn.getresponse()
data = res.read()

response_json = json.loads(data.decode("utf-8"))
output_path = Path(__file__).with_name("models.json")

with output_path.open("w", encoding="utf-8") as f:
   json.dump(response_json, f, indent=2)

print(f"Saved response to: {output_path}")