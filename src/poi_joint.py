## Join static information to the fitlered realtime feeds
## create stop pairs

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def prepare_stop_pairs(db_path,gap,ob_window):

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 1. join route and line information
        # First should clear out all indexes in case the database is locked
        df_index = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_auto%';",
            conn
        )
        for index_name in df_index['name']:
            cursor.execute(f'DROP INDEX IF EXISTS "{index_name}";')

        cursor.execute("""
        DROP TABLE IF EXISTS TripUpdates_poi_joint;
        """)
        # add historical dwell density for each stop i, in ob_window (second)
        # counted as, ob_window seconds before the scheduled time of arr_i, how many buses have passed by
        query = f"""        
            WITH observation AS (
                SELECT 
                    a.stop_id,
                    a.start_date,
                    a.trip_id,
                    a.arr_time,
                    COUNT(b.arr_time) AS his_dwell_count,
                    AVG(b.arr_delay) AS his_avg_delay,
                    AVG(b.duration) AS his_avg_dwell
                FROM tripupdates_poi a
                JOIN tripupdates_poi b
                    ON a.stop_id = b.stop_id
                    AND a.start_date = b.start_date
                    AND b.vehicle_id != a.vehicle_id
                    AND b.arr_time BETWEEN (a.arr_time - a.arr_delay - {ob_window})
                                    AND (a.arr_time - a.arr_delay)
                GROUP BY a.stop_id, a.start_date, a.trip_id, a.arr_time
            )
            SELECT 
                tp.*,
                r.route_id,
                r.route_short_name AS line_name,
                r.route_type,
                COALESCE(obs.his_dwell_count, 0) AS his_dwell_count,
                COALESCE(obs.his_avg_delay, 0) AS his_avg_delay,
                COALESCE(obs.his_avg_dwell, 0) AS his_avg_dwell
            FROM tripupdates_poi tp
            JOIN trips t ON tp.trip_id = t.trip_id
            JOIN routes r ON r.route_id = t.route_id
            LEFT JOIN observation obs
                ON tp.stop_id = obs.stop_id 
                AND tp.start_date = obs.start_date 
                AND tp.trip_id = obs.trip_id
                AND tp.arr_time = obs.arr_time;
            """
        df = pd.read_sql_query(query, conn)
        df.to_sql("TripUpdates_poi_joint", conn, if_exists="replace", index=False)


    
    # 2. prepare stop pairs

        cursor.execute("DROP TABLE IF EXISTS stop_pairs")
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tripupdates_trip_vehicle_date_seq
        ON TripUpdates_poi(trip_id, vehicle_id, start_date, stop_sequence);
        """)
        print("created index for (trip_id, vehicle_id, start_date, stop_sequence)")
        pair_query = f"""
        CREATE TABLE stop_pairs AS
        SELECT 
            a.trip_id as trip_id,
            a.start_date as start_date,
            a.vehicle_id as vehicle_id,
            a.route_id as route_id,
            a.line_name as line_name,
            a.route_type as route_type,
            a.stop_id as stop_i,
            a.stop_sequence as seq_i,
            a.arr_delay as arr_delay_i,
            a.dep_delay as dep_delay_i,
            a.arr_time as arr_time_i,
            a.duration as duration_i,
            a.his_dwell_count as his_dwell_count_i,
            a.his_avg_delay as his_avg_delay_i,
            a.his_avg_dwell as his_avg_dwell_i,
            b.stop_id as stop_j,
            b.stop_sequence as seq_j,
            b.arr_delay as arr_delay_j,
            b.dep_delay as dep_delay_j,
            b.arr_time as arr_time_j,
            b.duration as duration_j,
            (b.arr_time - a.arr_time) as travel_time,
            (b.arr_delay - a.arr_delay) as delay_change,
            CAST(strftime('%H', datetime(a.arr_time, 'unixepoch')) AS INTEGER) as hour_of_day,
            CAST(strftime('%w', substr(a.start_date,1,4) || '-' || substr(a.start_date,5,2) || '-' || substr(a.start_date,7,2)) AS INTEGER) AS day_of_week
        FROM TripUpdates_poi_joint a
        JOIN TripUpdates_poi_joint b 
            ON a.trip_id = b.trip_id 
            AND a.vehicle_id = b.vehicle_id
            AND a.start_date = b.start_date
            AND b.stop_sequence = a.stop_sequence+{gap}
        """

        cursor.execute(pair_query)


def print_basic_statistics(db_path):
    with sqlite3.connect(db_path) as conn:
        # Basic statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT trip_id) as n_trips,
            COUNT(DISTINCT stop_id) as n_stops,
            COUNT(DISTINCT start_date) as n_days,
            MIN(start_date) as first_date,
            MAX(start_date) as last_date
        FROM TripUpdates_poi
        """
        stats = pd.read_sql(stats_query, conn)
        print(stats.to_string(index=False))
