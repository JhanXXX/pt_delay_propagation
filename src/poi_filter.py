## filter out the records in interested area
## remove duplicated records (take the latest update)


import sqlite3
import pandas as pd

def filter_stop_poi(db_path):
    """Filter interested stops within Stockholm"""

    # Decided by hand: approximately Stockholm urban area, approximately between Nacka and Solna
    lat_min = 59.301280
    lat_max = 59.384970
    lon_min = 18.132843
    lon_max = 18.955033
   
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Drop and recreate stops_poi table
        # First should clear out all indexes in case the database is locked
        df_index = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_auto%';",
            conn
        )
        for index_name in df_index['name']:
            cursor.execute(f'DROP INDEX IF EXISTS "{index_name}";')

        cursor.execute("DROP TABLE IF EXISTS stops_poi")
        
        cursor.execute("""
        CREATE TABLE stops_poi (
            stop_id TEXT PRIMARY KEY,
            stop_name TEXT,
            stop_lat REAL,
            stop_lon REAL
        )
        """)
        
        # Insert filtered stops
        cursor.execute(f"""
        INSERT INTO stops_poi
        SELECT DISTINCT stop_id, stop_name, stop_lat, stop_lon
        FROM stops
        WHERE stop_lat BETWEEN {lat_min} AND {lat_max}
          AND stop_lon BETWEEN {lon_min} AND {lon_max}
        """)
        


def filter_trip_updates_poi(db_path):
    """Filter trip updates to only include stops in POI area"""
    """Filter out the duplicated udpates and only preserve the latest udpate"""
    # First ensure stops_poi exists
    filter_stop_poi(db_path)
    
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        cursor = conn.cursor()
        
        # Drop existing table
        cursor.execute("DROP TABLE IF EXISTS TripUpdates_poi")
        
        # Create filtered trip updates table
        cursor.execute("""
            CREATE TABLE TripUpdates_poi AS
            SELECT tu.*
            FROM TripUpdates tu
            WHERE tu.trip_id IS NOT NULL
            AND tu.trip_id != ''
            AND tu.stop_id IN (SELECT stop_id FROM stops_poi)
        """)

        result = cursor.execute("SELECT COUNT(*) FROM TripUpdates_poi").fetchone()
        print(f"Filtered to {result[0]} trip updates in POI area")

        
        # Remove duplicates - keep the latest update for each unique trip arrival
        cursor.execute("""
            DELETE FROM TripUpdates_poi
            WHERE rowid NOT IN (
                SELECT rowid FROM (
                    SELECT 
                        rowid,
                        ROW_NUMBER() OVER (
                            PARTITION BY vehicle_id, trip_id, schedule_relationship, 
                                        start_date, stop_sequence, stop_id
                            ORDER BY update_timestamp DESC
                        ) AS rn
                    FROM TripUpdates_poi
                )
                WHERE rn = 1
            );
        """)
        
        # Check results
        result = cursor.execute("SELECT COUNT(*) FROM TripUpdates_poi").fetchone()
        print(f"Filtered to {result[0]} trip updates in POI area")