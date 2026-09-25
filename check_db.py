import sqlite3, os

db = 'users.db'
print('DB exists:', os.path.exists(db))
conn = sqlite3.connect(db)
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', c.fetchall())
c.execute("PRAGMA table_info(users)")
print('Columns:', c.fetchall())
conn.close()
