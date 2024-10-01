import sqlite3
from crawl import run_prompt

def config_database():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            user_email TEXT NOT NULL,
            user_name TEXT NOT NULL
        )
    ''')

def main():
    config_database()
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    userName = input("Your name: ")
    userEmail = input("Your email: ")
    prompt = input("Your prompt: ")
    c.execute("INSERT INTO prompts (prompt, user_email, user_name) VALUES (?, ?, ?)", (prompt, userEmail, userName))
    conn.commit()
    final_result =run_prompt(prompt)
    print(final_result)

  
  
if __name__ == '__main__':
    main()