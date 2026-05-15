import sqlite3
import os
import hashlib


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'closet.db')
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS closet_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    image_path TEXT,
                    ana_kategori TEXT NOT NULL,
                    alt_kategori TEXT NOT NULL,
                    marka TEXT,
                    beden TEXT,
                    renk TEXT,
                    desen TEXT,
                    materyal TEXT,
                    kalip TEXT,
                    boy TEXT,
                    bel_tipi TEXT,
                    paca_tipi TEXT,
                    paca_boyu TEXT,
                    kol_tipi TEXT,
                    yaka_tipi TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')

    conn.commit()
    conn.close()


def _hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def create_user(email, password):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)",
                  (email, _hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(email, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ? AND password_hash = ?",
              (email, _hash_password(password)))
    result = c.fetchone()
    conn.close()
    return result is not None


def user_exists(email):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    return result is not None


def save_item(data):
    conn = get_connection()
    c = conn.cursor()
    query = """INSERT INTO closet_items
               (username, image_path, ana_kategori, alt_kategori, marka, beden, renk,
                desen, materyal, kalip, boy, bel_tipi, paca_tipi, paca_boyu, kol_tipi, yaka_tipi)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    params = (
        data.get('username'), data.get('image_path'), data.get('ana_kategori'),
        data.get('alt_kategori'), data.get('marka'), data.get('beden'),
        data.get('renk'), data.get('desen'), data.get('materyal'),
        data.get('kalip'), data.get('boy'), data.get('bel_tipi'),
        data.get('paca_tipi'), data.get('paca_boyu'), data.get('kol_tipi'), data.get('yaka_tipi')
    )
    c.execute(query, params)
    conn.commit()
    conn.close()


def get_user_items(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM closet_items WHERE username = ? ORDER BY created_at DESC", (username,))
    items = [dict(row) for row in c.fetchall()]
    conn.close()
    return items


def delete_item(item_id, username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT image_path FROM closet_items WHERE id = ? AND username = ?",
              (item_id, username))
    row = c.fetchone()
    if row and row['image_path'] and os.path.exists(row['image_path']):
        os.remove(row['image_path'])
    c.execute("DELETE FROM closet_items WHERE id = ? AND username = ?", (item_id, username))
    conn.commit()
    conn.close()


def save_uploaded_image(username, uploaded_file):
    user_dir = os.path.join(UPLOADS_DIR, username.replace('@', '_').replace('.', '_'))
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path
