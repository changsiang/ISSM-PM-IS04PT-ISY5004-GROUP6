import io
import numpy as np
import sqlite3
from sqlite3 import Error

class DatabaseService:
    __version__ = '0.0.1'
    db_path = "database.db"
    conn = None

    def adapt_array(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    
    def createConnection(self):
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        try:
            self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        except Error as e:
            print(e)

    def create_table(self, query):
        try:
            c = self.conn.cursor()
            c.execute(query)
        except Error as e:
            print(e)
        finally:
            c.close()
    
    def initializePatientTable(self):
        query = """ CREATE TABLE IF NOT EXISTS patient (
            id integer PRIMARY KEY AUTOINCREMENT,
            name text NOT NULL,
            feature array NOT NULL
        ); """
        self.create_table(query)                                                                                                                                                                                                

    def insertPatientRow(self, patient_name, features):
        query = "INSERT INTO patient(name, feature) VALUES (?, ?)"
        rowCount = self.getPatientNameCount(patient_name)
        if rowCount is not None and rowCount > 5:
            return
        else:
            try:
                c = self.conn.cursor()
                c.execute(query, (patient_name, features, ))
                self.conn.commit()
            except Error as e:
                print("insert row", e)
            finally:
                c.close()

    def getPatientNameCount(self, patient_name):
        query = "SELECT COUNT(*) FROM patient WHERE name = ?"
        c = self.conn.cursor()
        try:
            c.execute(query, (patient_name,))
            result = c.fetchone()
            return result[0]
        except Error as e:
            print(e)
        finally:
            c.close()

    def getAllPatients(self):
        try:
            c = self.conn.cursor()
            c.execute("SELECT * FROM patient")
            rows = c.fetchall()
            return rows # list of tuples
        except Error as e:
            print(e)
        finally:
            c.close()