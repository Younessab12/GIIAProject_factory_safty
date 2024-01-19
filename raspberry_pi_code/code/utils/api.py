import requests
import time
import json

class API:
  def __init__(self, baseUrl, apiKey):
    self.baseUrl = baseUrl
    self.apiKey = apiKey

  def ping(self, operatorName):
    payload = {
      'apiKey': self.apiKey,
      'operatorName': operatorName
    }
    url = self.baseUrl + '/api/ping'
    return requests.post(url, json=payload)

  def reportActivity(self, activity):
    """
      apiKey: z.string(),
    x  operatorName: z.string(),
    x  gravity: z.number(),
    x  activityName: z.string(),
      timestamp: z.string(),
    """
    timestamp = int(time.time())
    print(timestamp)
    payload = {
      'apiKey': self.apiKey,
      'operatorName': activity['operatorName'],
      'gravity': activity['gravity'],
      'activityName': activity['activityName'],
      'timestamp': timestamp*1000+3600*1000
    }
    url = self.baseUrl + '/api/reportActivity'
    return requests.post(url, json=payload)
  