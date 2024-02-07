#!/usr/bin/env python3
import os
import requests
import json
import re
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import argparse
import sys

BASE_URL = "https://api.commadotai.com/v1"
HOME_DIR = os.path.expanduser("~")
access_token = ""

try:
    with open(os.path.join(HOME_DIR, ".comma/auth.json"), "r") as f:
        access_token = json.loads(f.read())["access_token"]
except IOError as e:
    print(f"Auth file could not be read: {e}")
    exit(1)


def unix_time_millis(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000.0)


def routes_segments(device, start, end):
    url = f"{BASE_URL}/devices/{device}/routes_segments?start={unix_time_millis(start)}"
    if end is not None:
        url += f"&end={unix_time_millis(end)}"
    response = requests.get(url, headers={"Authorization": f"JWT {access_token}"})
    if response.ok:
        return response.json()
    else:
        print(f"Could not get routes/segments:\n{response.text}")
        return None


def route_files(route_name):
    # /v1/route/:route_name/files
    url = f"{BASE_URL}/route/{route_name}/files"
    response = requests.get(url, headers={"Authorization": f"JWT {access_token}"})
    if response.ok:
        return response.json()
    else:
        print(f"Could not get route files:\n{response.text}")
        print(f"Tried url: {url}")
        return None


def parse_route_file_url(url):
    regex = "(\/([\w\d]+)\/([\d\-]+)\/(\d+)\/([\w\d\.]+))"
    match = re.search(regex, url)
    if match is not None:
        groups = match.groups()
        assert (len(groups) == 5)
        return groups[1], groups[2], groups[3], groups[4]
    return None


def download_route(route_name, data_dir):
    # Ensure dir exists
    if os.path.exists(data_dir) is False:
        print("Data dir does not exist")
        return

    # Get route files
    files_descriptor = route_files(route_name)
    if files_descriptor is None:
        return

    for type_name, files in files_descriptor.items():
        print(f"Processing {type_name}")
        for file in files:
            device_id, route_id, segment_id, file_name = parse_route_file_url(file)
            segment_dir = os.path.join(data_dir, device_id, f"{route_id}--{segment_id}")
            os.makedirs(segment_dir, exist_ok=True)
            output_file = os.path.join(segment_dir, file_name)
            print(f"Downloading file: {file} to {output_file}")
            r = requests.get(file, allow_redirects=True)
            with open(output_file, 'wb') as f:
                f.write(r.content)


def route(route_name):
    response = requests.get(f"{BASE_URL}/route/{route_name}/segments", headers={"Authorization": f"JWT {access_token}"})
    return response.json() if response.ok else {}


def me():
    response = requests.get(f"{BASE_URL}/me", headers={"Authorization": f"JWT {access_token}"})
    return response.json() if response.ok else {}


def devices():
    response = requests.get(f"{BASE_URL}/me/devices/", headers={"Authorization": f"JWT {access_token}"})
    return response.json() if response.ok else []


# Main command parser
mainArgParser = argparse.ArgumentParser(prog='comma_connect_client', description='Interact with connect.comma.ai')
mainArgParser.add_argument('command', choices=['me', 'devices', 'routes', 'route', 'download'])
args, extra = mainArgParser.parse_known_args()


def parse_date_arg(input, date_format='%Y-%m-%d'):
    return datetime.strptime(input, date_format)


match args.command:
    case 'me':
        print(json.dumps(me()))
    case 'devices':
        result = devices()
        print(json.dumps(result))
    case 'routes':
        parser = argparse.ArgumentParser(prog='comma_connect_client routes', description='Get routes for a device')
        parser.add_argument("--device", required=True)
        parser.add_argument("--start", required=True, type=parse_date_arg)
        parser.add_argument("--end", required=True, type=parse_date_arg)
        args = parser.parse_args(args=extra)
        result = routes_segments(args.device, args.start, args.end)
        print(json.dumps(result))
    case 'route':
        parser = argparse.ArgumentParser(prog='comma_connect_client route',
                                         description='Get info on a route')
        parser.add_argument("--route", required=True)
        args = parser.parse_args(args=extra)
        print(json.dumps(route(args.route)))
    case 'download':
        parser = argparse.ArgumentParser(prog='comma_connect_client download',
                                         description='Download files for a route')
        parser.add_argument("--route", required=True)
        parser.add_argument("--dir", required=True)
        args = parser.parse_args(args=extra)
        download_route(args.route, args.dir)
    case _:
        print("RTFM")
