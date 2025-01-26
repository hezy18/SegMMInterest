import psutil
import os

def get_command():
    my_process = psutil.Process( os.getpid() )
    return " ".join(my_process.cmdline())