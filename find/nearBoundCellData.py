import spacemap
import numpy as np
import pandas as pd
import json
import os

import sqlite3

class CacheKVDB:
    def __init__(self, db_path):
       self.db_path = db_path
       self.DB = {}
       spacemap.mkdir(db_path)
       
    def auto_init(self, path):
        path1 = self.db_path + "/%s.json" % path
        if path in self.DB:
            return
        if os.path.exists(path1):
            with open(path1, "r") as f:
                self.DB[path] = json.load(f)
        else:
            self.DB[path] = {}
    
    def insert_kv(self, cellname, path, value):
        self.auto_init(path)
        self.DB[path][cellname] = value
    
    def query_value(self, cellname, path):
        self.auto_init(path)
        return self.DB[path].get(cellname, None)

    def close(self):
        for key in self.DB.keys():
            path1 = self.db_path + "/%s.json" % key
            with open(path1, "w") as f:
                json.dump(self.DB[key], f)

class SimpleKVDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS kv_store (
                key1 TEXT,
                key2 TEXT,
                value TEXT,
                PRIMARY KEY (key1, key2)
            )
        ''')
        # 创建复合索引以优化查询性能
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_keys ON kv_store (key1, key2)
        ''')
        self.conn.commit()
    
    def insert_kv(self, key1, key2, value):
        try:
            self.cursor.execute("INSERT INTO kv_store (key1, key2, value) VALUES (?, ?, ?) ON CONFLICT(key1, key2) DO UPDATE SET value = excluded.value", (key1, key2, value))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
    
    def query_value(self, key1, key2):
        self.cursor.execute("SELECT value FROM kv_store WHERE key1 = ? AND key2 = ?", (key1, key2))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()


class NearBoundCellData:
    def __init__(self, db: SimpleKVDB, name):
        self.name = name
        self.db = db
        
    def save(self, key, data, typ):
        path = "%s_%s" % (key, typ)
        if typ == "np":
            data = data.tolist()
            self.db.insert_kv(self.name, path, json.dumps(data))
        elif typ == "lis":
            self.db.insert_kv(self.name, path, json.dumps(data))
        elif typ == "df":
            data = data.to_json(orient="records")
            self.db.insert_kv(self.name, path, data)
        elif typ == "num":
            self.db.insert_kv(self.name, path, str(data))
        else:
            raise Exception("Type Error")
    
    def load(self, key, typ):
        path = "%s_%s" % (key, typ)
        v = self.db.query_value(self.name, path)
        if v is None:
            return None
        if typ == "np":
            return np.array(json.loads(v))
        if typ == "lis":
            return json.loads(v)
        if typ == "df":
            return pd.read_json(v)
        if typ == "num":
            return float(v)
        raise Exception("Type Error")
        return v
        