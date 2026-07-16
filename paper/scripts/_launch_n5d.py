import subprocess
p = subprocess.Popen(
    ['D:\\Miniconda3\\python.exe', '-u', r'C:\Users\高帅东\Desktop\causalscale\paper\scripts\run_n5d.py'],
    stdout=open(r'D:\NO.1\causalscale_kdd2027_experiments\n5d_run.log', 'w'),
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NO_WINDOW
)
print(f'n=5d experiment PID: {p.pid}')
