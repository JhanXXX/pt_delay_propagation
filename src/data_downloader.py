## download data from gtfs: realtime tripupdates and static trips information
## data managed using sqlite3
## the database is loosely designed. data structure can be seen in the coding
## there are other static information including stops, calendars etc., which most only requires one-step action
## I didnt put the corresponding code here, but they are all named as GTFS standard.

import os
import requests
import py7zr
import sqlite3
from pathlib import Path
import pandas as pd
import zipfile
from google.transit import gtfs_realtime_pb2
import gtfs_kit as gk
import time

def create_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.close()


def prepare_sl_gtfs_realtime(api_key,year,month,day,hour):
    feed_name = "TripUpdates" 
    # hour = "09"
    # date = "2025-03-11"
    date = f"{year}-{month}-{day}"
    operator = "sl"

    url = f"https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/{operator}/{feed_name}?date={date}&hour={hour}&key={api_key}"
    save_path = f"workspace/data/GTFS_RT"
    extract_path = f"workspace/data/GTFS_RT_extracted"
    db_root = "workspace/database"
    db_path = f"{db_root}/gtfs_raw.db"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(db_root, exist_ok=True)
    create_db(db_path)
    
    # 1. download
    downloaded_pd = f"{save_path}/{operator}_{date}_{hour}_{feed_name}.pb"
    if not os.path.exists(downloaded_pd):
        response = requests.get(url)
        if response.status_code == 200:
            with open(downloaded_pd, "wb") as f:
                f.write(response.content)
            print(f"{date} download suceeded!")
        else:
            print(f"failed: {date}.", response.status_code)
            return # if no data available, skip
    time.sleep(5)
    # 2. extract
    with py7zr.SevenZipFile(downloaded_pd, mode='r') as archive:
        archive.extractall(path=extract_path)

    # 3. connect to the sqlite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        ## initialize tables if not exist
        init_query_entity = f"""
            CREATE TABLE IF NOT EXISTS {feed_name} (
                entity_id TEXT,
                vehicle_id TEXT,
                trip_id TEXT,
                schedule_relationship INTEGER,
                start_date TEXT,
                update_timestamp INTEGER,
                stop_sequence INTEGER,
                stop_id TEXT,
                arr_time INTEGER,
                arr_delay INTEGER,
                dep_time INTEGER,
                dep_delay INTEGER,
                duration INTEGER
            )
        """

        cursor.execute(init_query_entity)


        # 4. get all .pb files under extract path and insert the raw data into database
        extract_root = Path(f"{extract_path}/{operator}/{feed_name}/{year}/{month}/{day}/{hour}")
        pb_files = list(extract_root.rglob("*.pb")) 

        for pb_file in pb_files:
            feed = gtfs_realtime_pb2.FeedMessage()
            with open(pb_file, "rb") as f:
                data = f.read()
                feed.ParseFromString(data)
            
            for entity in feed.entity:
                entity_id = entity.id
                vehicle_id = entity.trip_update.vehicle.id
                if entity.trip_update.trip.HasField("trip_id"):
                    trip_id = entity.trip_update.trip.trip_id
                    schedule_relationship = entity.trip_update.trip.schedule_relationship
                    start_date = entity.trip_update.trip.start_date
                    update_timestamp = entity.trip_update.timestamp
                    for situ in entity.trip_update.stop_time_update:
                        if situ.arrival.HasField("uncertainty") and situ.arrival.uncertainty==0: 
                            stop_sequence = situ.stop_sequence
                            stop_id = situ.stop_id
                            arr_time = situ.arrival.time
                            arr_delay = situ.arrival.delay
                            dep_time = situ.departure.time
                            dep_delay = situ.departure.delay
                            duration = dep_time - arr_time
                            ### insert operation
                            cursor.execute(f"""
                                INSERT OR IGNORE INTO {feed_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (entity_id, vehicle_id, trip_id, schedule_relationship, 
                                start_date, update_timestamp, stop_sequence, stop_id,
                                arr_time, arr_delay, dep_time, dep_delay, duration))


def prepare_sl_gtfs_static(api_key,year,month,day):
    # date = "2025-03-11"
    date = f"{year}-{month}-{day}"
    operator = "sl"

    url = f"https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/{operator}?date={date}&key={api_key}"
    save_path = f"workspace/data/GTFS_static"
    extract_path = f"workspace/data/GTFS_static/extracted/{date}"
    db_root = "workspace/database"
    db_path = f"{db_root}/gtfs_raw.db"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(db_root, exist_ok=True)
    create_db(db_path)

    # 1. download if not exist
    downloaded_zip = f"{save_path}/{operator}_{date}.zip"
    if not os.path.exists(downloaded_zip):
        response = requests.get(url)
        if response.status_code == 200:
            with open(downloaded_zip, "wb") as f:
                f.write(response.content)
            print(f"download {date} suceeded!")
        else:
            print("failed: ", response.status_code)
            return # if no data available, skip
        
        time.sleep(5)

    # 2. extract (not mandatory)
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(downloaded_zip, 'r') as archive:
            archive.extractall(path=extract_path)
    
    # 3. connect to the database 
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        ## initialize tables if not exist
        init_query_trips = f"""
            CREATE TABLE IF NOT EXISTS trips (
                route_id TEXT,
                trip_id TEXT,
                direction_id INTEGER
            )
        """
        cursor.execute(init_query_trips)

        # insert trips and stop_times
        feed_path = downloaded_zip
        feed = gk.read_feed(feed_path, dist_units='km')

        df_trips = feed.trips
        trips_data = df_trips[['route_id','trip_id','direction_id']].values.tolist()
        cursor.executemany(
            "INSERT OR IGNORE INTO trips (route_id, trip_id, direction_id) VALUES (?, ?, ?)",
            trips_data
        )
        cursor.execute("""
            DELETE FROM trips
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM trips
                GROUP BY route_id, trip_id, direction_id
            )
            """)
        
    
