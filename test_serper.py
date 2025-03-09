import http.client
import json

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({
  "q": "apple inc"
})
headers = {
  'X-API-KEY': 'b2af04d2149e997c5b923932db9a5c7e96bfb9ef',
  'Content-Type': 'application/json'
}
conn.request("POST", "/search", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))