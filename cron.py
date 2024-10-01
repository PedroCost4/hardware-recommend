from datetime import datetime
from croniter import croniter
import time
import dotenv
from crawl import run_crawler, configure_settings
import os

dotenv.load_dotenv()

testing = os.getenv("TESTING") == "True"
expressao_cron = "0 8 * * *" if testing else "*/5 * * * *"

configure_settings()

while True:
    agendador = croniter(expressao_cron, datetime.now())
    proximo_agendamento = agendador.get_next(datetime)

    espera = proximo_agendamento - datetime.now()
    print("esperando pela proxima execução:", espera)
    # time.sleep(espera.total_seconds())

    run_crawler(None)