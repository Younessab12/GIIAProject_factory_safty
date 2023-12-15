import os
import datetime

class Logger:
  def __init__(self, log_file):
    self.log_file = log_file
    self.log_file_path = None

  def __enter__(self):
    self.open_log_file()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close_log_file()

  def open_log_file(self):
    logs_folder = os.path.join(os.path.dirname(self.log_file), "logs")
    os.makedirs(logs_folder, exist_ok=True)

    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Create a log file for the current date
    self.log_file_path = os.path.join(logs_folder, f"log_{current_date}.txt")

    self.file = open(self.log_file_path, "a")

  def close_log_file(self):
    if self.file:
      self.file.close()

  def log(self, metadata):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {metadata}\n"

    if not self.file:
      self.open_log_file()

    self.file.write(log_entry)

class LogExtractor:
  def __init__(self, log_file):
    self.log_file = log_file

  def extract_logs(self):
    logs = []
    with open(self.log_file, "r") as file:
      lines = file.readlines()
      for line in lines:
        timestamp, metadata = line.strip().split(" - ")
        log = {"timestamp": timestamp, "metadata": metadata}
        logs.append(log)
    return logs
