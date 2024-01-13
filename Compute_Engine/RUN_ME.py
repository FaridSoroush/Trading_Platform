# pip install psutil
import psutil
import subprocess
import time

name_of_the_main_script = "Deployment_main.py"
# check everying 5 minutes:
checking_time = 60 * 3

def check_if_process_running(process_name):
    '''Check if there is any running process that contains the given name.'''
    for proc in psutil.process_iter(['name']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def run_process(process_path):
    '''Run the given process.'''
    process = subprocess.Popen(["python3", process_path], shell=False)
    process.wait()


print("file excuted")

while True:
    # Check every 10 minutes
    if not check_if_process_running("temp_printer.py"):
        run_process("/Users/faridsoroush/Documents/GitHub/Trading-Software/Compute_Engine/Deployment_main.py")

    time.sleep(checking_time) 
