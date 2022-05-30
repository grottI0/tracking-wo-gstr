import os

from add_to_db import add_to_db

import requests
import subprocess
import time
import signal


def test_server_200():
    a = subprocess.Popen('python3 -m http.server 9000', shell=True, preexec_fn=os.setsid)
    time.sleep(1)
    res = requests.get('http://localhost:9000/')
    os.killpg(os.getpgid(a.pid), signal.SIGTERM)
    assert res.status_code == 200


def test_server_404():
    a = subprocess.Popen('python3 -m http.server 9000', shell=True, preexec_fn=os.setsid)
    time.sleep(1)
    res = requests.get('http://localhost:9000/smth')
    os.killpg(os.getpgid(a.pid), signal.SIGTERM)
    assert res.status_code == 404


def test_tracking():
    result = subprocess.run('python track.py --source videos/test.mp4 --yolo_model weights/best_final.pt', shell=True,
                            stdout=subprocess.PIPE, encoding='utf-8')
    assert result.returncode == 0


def test_insert_1():
    inf = {'name1': 'fight', 'accuracy1': 13.397889999999999, 'time1': 16, 'time_ent_1': '10:57:08',
           'time_out_1': '10:57:19', 'name2': 'fight', 'accuracy2': 74.86325999999995, 'time2': 91,
           'time_ent_2': '10:57:08', 'time_out_2': '10:58:02', 'name3': 'fight', 'accuracy3': 1.45782, 'time3': 2,
           'time_ent_3': '10:57:21', 'time_out_3': '10:57:21'}
    all_id = [1, 2, 3]
    assert add_to_db(inf, all_id)[0] == ['1', 'fight', '10:57:08', '10:57:19', '0.84']


def test_insert_2():
    inf = {'name1': 'fight', 'accuracy1': 13.397889999999999, 'time1': 16, 'time_ent_1': '10:57:08',
           'time_out_1': '10:57:19', 'name2': 'fight', 'accuracy2': 74.86325999999995, 'time2': 91,
           'time_ent_2': '10:57:08', 'time_out_2': '10:58:02', 'name3': 'fight', 'accuracy3': 1.45782, 'time3': 2,
           'time_ent_3': '10:57:21', 'time_out_3': '10:57:21'}
    all_id = [1, 2, 3]
    assert add_to_db(inf, all_id)[1] == ['2', 'fight', '10:57:08', '10:58:02', '0.82']


def test_insert_3():
    inf = {'name1': 'fight', 'accuracy1': 13.397889999999999, 'time1': 16, 'time_ent_1': '10:57:08',
           'time_out_1': '10:57:19', 'name2': 'fight', 'accuracy2': 74.86325999999995, 'time2': 91,
           'time_ent_2': '10:57:08', 'time_out_2': '10:58:02', 'name3': 'fight', 'accuracy3': 1.45782, 'time3': 2,
           'time_ent_3': '10:57:21', 'time_out_3': '10:57:21', 'name4': 'fight', 'accuracy4': 6.755979999999999,
           'time4': 9, 'time_ent_4': '10:57:24', 'time_out_4': '10:57:29', 'name7': 'fight',
           'accuracy7': 15.783180000000003, 'time7': 21, 'time_ent_7': '10:57:41', 'time_out_7': '10:57:54'}
    all_id = [1, 2, 3, 4, 7]
    assert add_to_db(inf, all_id)[4] == ['7', 'fight', '10:57:41', '10:57:54', '0.75']


def test_insert_4():
    inf = {'name1': 'fight', 'accuracy1': 13.397889999999999, 'time1': 16, 'time_ent_1': '10:57:08',
           'time_out_1': '10:57:19', 'name2': 'fight', 'accuracy2': 74.86325999999995, 'time2': 91,
           'time_ent_2': '10:57:08', 'time_out_2': '10:58:02', 'name3': 'fight', 'accuracy3': 1.45782, 'time3': 2,
           'time_ent_3': '10:57:21', 'time_out_3': '10:57:21', 'name4': 'fight', 'accuracy4': 6.755979999999999,
           'time4': 9, 'time_ent_4': '10:57:24', 'time_out_4': '10:57:29', 'name7': 'fight',
           'accuracy7': 15.783180000000003, 'time7': 21, 'time_ent_7': '10:57:41', 'time_out_7': '10:57:54',
           'name9': 'fight', 'accuracy9': 19.90835, 'time9': 26, 'time_ent_9': '10:58:00', 'time_out_9': '10:58:16',
           'name11': 'fight', 'accuracy11': 53.42228000000002, 'time11': 70, 'time_ent_11': '10:58:07',
           'time_out_11': '10:58:45', 'name12': 'fight', 'accuracy12': 1.3086600000000002, 'time12': 2,
           'time_ent_12': '10:58:27', 'time_out_12': '10:58:28', 'name13': 'fight', 'accuracy13': 7.4717899999999995,
           'time13': 10, 'time_ent_13': '10:58:33', 'time_out_13': '10:58:39', 'name15': 'fight', 'accuracy15': 2.15566,
           'time15': 3, 'time_ent_15': '10:58:44', 'time_out_15': '10:58:45'}
    all_id = [1, 2, 3, 4, 7, 9, 11, 12, 13, 15]
    assert add_to_db(inf, all_id)[8] == ['13', 'fight', '10:58:33', '10:58:39', '0.75']


def test_insert_5():
    inf = {'name1': 'fight', 'accuracy1': 13.397889999999999, 'time1': 16, 'time_ent_1': '10:57:08',
           'time_out_1': '10:57:19', 'name2': 'fight', 'accuracy2': 74.86325999999995, 'time2': 91,
           'time_ent_2': '10:57:08', 'time_out_2': '10:58:02', 'name3': 'fight', 'accuracy3': 1.45782, 'time3': 2,
           'time_ent_3': '10:57:21', 'time_out_3': '10:57:21', 'name4': 'fight', 'accuracy4': 6.755979999999999,
           'time4': 9, 'time_ent_4': '10:57:24', 'time_out_4': '10:57:29', 'name7': 'fight',
           'accuracy7': 15.783180000000003, 'time7': 21, 'time_ent_7': '10:57:41', 'time_out_7': '10:57:54',
           'name9': 'fight', 'accuracy9': 19.90835, 'time9': 26, 'time_ent_9': '10:58:00', 'time_out_9': '10:58:16',
           'name11': 'fight', 'accuracy11': 53.42228000000002, 'time11': 70, 'time_ent_11': '10:58:07',
           'time_out_11': '10:58:45', 'name12': 'fight', 'accuracy12': 1.3086600000000002, 'time12': 2,
           'time_ent_12': '10:58:27', 'time_out_12': '10:58:28', 'name13': 'fight', 'accuracy13': 7.4717899999999995,
           'time13': 10, 'time_ent_13': '10:58:33', 'time_out_13': '10:58:39', 'name15': 'fight', 'accuracy15': 2.15566,
           'time15': 3, 'time_ent_15': '10:58:44', 'time_out_15': '10:58:45'}
    all_id = [1, 2, 3, 4, 7, 9, 11, 12, 13, 15]
    assert add_to_db(inf, all_id)[9] == ['15', 'fight', '10:58:44', '10:58:45', '0.72']
