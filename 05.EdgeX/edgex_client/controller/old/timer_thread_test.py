from threading import Timer, Thread, Event
from datetime import datetime


def ten_ms_timer():
    tempo = datetime.utcnow().strftime('%H-%M-%S.%f')[:-3]
    print(tempo)
    Timer(0.01, ten_ms_timer).start()

ten_ms_timer()