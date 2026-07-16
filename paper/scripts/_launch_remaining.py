import subprocess
p = subprocess.Popen(
    ['D:\\Miniconda3\\python.exe', '-u', r'C:\Users\高帅东\Desktop\causalscale\paper\scripts\run_remaining.py'],
    stdout=open(r'D:\NO.1\causalscale_kdd2027_experiments\remaining_run.log', 'w'),
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NO_WINDOW
)
print(f'Remaining experiments PID: {p.pid}')
