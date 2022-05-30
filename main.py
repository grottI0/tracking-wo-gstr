import os
import time
import multiprocessing
import subprocess
import sys

call_type = 'User'

if len(sys.argv) > 1:
    if sys.argv[1].lower() == 'docker':
        call_type = 'Docker'


def server():
    os.chdir('for_server')
    while 'start.txt' not in os.listdir():
        time.sleep(1)
    subprocess.run('python3 -m http.server 9000', shell=True)


def tracking():
    if call_type == 'User':
        subprocess.run('python3 track.py --yolo_model weights/best_final.pt --source videos/fight_small.mp4 --save-vid', shell=True)
    elif call_type == 'Docker':
        subprocess.run('python3 track.py --docker --yolo_model weights/best_final.pt --source videos/fight_small.mp4 --save-vid', shell=True)


os.chdir('for_server')
directory = os.listdir()
try:
    os.system('rm out.webm')
except Exception:
    pass
for i in directory:
    if i != 'index.html':
        os.system(f'rm {i}')

os.chdir(os.pardir)

p1 = multiprocessing.Process(target=tracking)
p2 = multiprocessing.Process(target=server)

p2.start()
p1.start()
