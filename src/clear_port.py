from psutil import process_iter
from signal import SIGTERM # or SIGKILL

def clear_port(port: int) -> None:
    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port == port:
                # print(conns)
                try:
                    proc.send_signal(SIGTERM) # or SIGKILL
                # When you rerun the app and it's already running on your server
                # dash will give a permission error and will stop the re-running
                # this just ignores that error and forces to keep trying to re-run
                except:
                    pass
