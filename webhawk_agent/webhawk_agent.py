import argparse
import requests
import socket
import configparser
import sys

config = configparser.ConfigParser()
config.sections()
config.read('settings.conf')

parser = argparse.ArgumentParser(description="Webhawk agent v1.0\nUse this script to make detection on your endpoints.")

parser.add_argument(
    "-l", 
    "--log_file",
    help = "Path to the Apache log file to scan (e.g., ./SAMPLE_DATA/RAW_APACHE_LOGS/access.log.2022-12-22)", 
    required = True
)

args = parser.parse_args()

try:
    with open(args.log_file,'r') as f:
        logs=f.read()
except FileNotFoundError:
    print(f"Error: The file {args.log_file} does not exist.")
    sys.exit(1)
except PermissionError:
    print(f"Error: Permission denied when accessing {args.log_file}.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error while reading the file: {e}")
    sys.exit(1)

data = {"hostname":socket.gethostname(),"logs_content":logs,"log_file":args.log_file}

try:
    print('Your logs are being processed..')
    response=requests.post("http://{}:{}/scan".format(config['WEBHAWK']['server_ip'],config['WEBHAWK']['server_port']),json=data)
    print(response.json())
except:
    print('Not able to reach webhawk service.\nMake sure that your webhawk server is up that that detection service is running.\nExiting..')
    sys.exit(1)