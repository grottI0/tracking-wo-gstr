import psycopg2
import time
import os

os.system('sudo -u postgres psql')

while True:
    try:
        conn = psycopg2.connect(dbname="yolo_db", user="postgres", password="123", host="localhost", port="5432")
        cursor = conn.cursor()
        print('Connected to database')
        break
    except:
        time.sleep(1)
        continue

try:
    cursor.execute("CREATE SCHEMA main;")
    print('Created schema')
except:
    print('Schema already exist')
    pass

try:
    cursor.execute("CREATE TABLE main.info (id SERIAl NOT NULL, number TEXT, fight_fall TEXT, time_ent TEXT,"
                   "time_out TEXT, average_accuracy TEXT);")
    print('Created table main.info')
    conn.commit()
except:
    print('Table already exist')
    pass