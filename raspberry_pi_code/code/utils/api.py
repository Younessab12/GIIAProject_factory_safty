import requests
import time
import json

class API:
  def __init__(self, baseUrl, apiKey):
    self.baseUrl = baseUrl
    self.apiKey = apiKey

  def reportActivity(self, activity):
    """
      apiKey: z.string(),
    x  operatorName: z.string(),
    x  gravity: z.number(),
    x  activityName: z.string(),
      timestamp: z.string(),
    """
    timestamp = int(time.time())
    payload = {
      'apiKey': self.apiKey,
      'operatorName': activity['operatorName'],
      'gravity': activity['gravity'],
      'activityName': activity['activityName'],
      'timestamp': timestamp
    }
    url = self.baseUrl + '/api/reportActivity'
    return requests.post(url, json=payload)
  