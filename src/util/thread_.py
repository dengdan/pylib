import threading
from threading import Thread

def get_current_thread():
    return threading.current_thread()
    
def is_alive(t):
    return t.is_alive()
    
def create_and_start(name, target, daemon = True):
    t = Thread(target= target)
    t.daemon = True
    t.setName(name)
    t.start()
    return t

