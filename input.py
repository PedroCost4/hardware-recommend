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
    
    print("")
    userName = input("Digite seu nome: ")
    userEmail = input("Digite seu email: ")
    prompt = input("Digite o prompt: ")

    c.execute("INSERT INTO prompts (prompt, user_email, user_name) VALUES (?, ?, ?)", (prompt, userEmail, userName))
    conn.commit()
    
    run_prompt(prompt)

  
  
if __name__ == '__main__':
    main()