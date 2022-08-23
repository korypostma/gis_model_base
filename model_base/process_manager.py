#NOTE to use:
#from model_base.process_manager import ProcessManager
#process_manager = ProcessManager(max_processes)
#def run_scripts(script, path_row, comp):
#    for i in range(0,max_processes-1+1):
#        process_number = str(i)
#        args = [path_row, comp, 'search', process_number]
#        #output_paths = []
#        process_manager.add(script + '_' + path_row + '_' + comp + '_' + process_number, script, args, output_paths=None)
#    exit_code = process_manager.start(script)
#    if exit_code != 0:
#        print('ERROR running:',script)
#        sys.exit(1)
#    return

from multiprocessing import Process #To fork new processes
import subprocess #In order to run command line scripts
import os #To get process ID
import time #To have delays and perf_counter
from datetime import timedelta #To show duration of runs
import sys #To allow exit codes to be passed to multiprocessing
from os.path import exists as file_exists #To check if files exist or not
import psutil

force_run = False #Set to True to ignore outputs if they already exist

#if os.name == 'nt':
#    base_path = 'D:/SMB3/shrubfs1'
#else:
#    base_path = '/caldera/projects/usgs/eros/rcmap'

max_processes = 20

def pm_safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        pass

def pm_run_script(script, args, log_file):
    if (not file_exists(script)):
        print(script,'script does not exist')
        sys.exit(-1)
    print(os.getppid(),'/',os.getpid(),'Running Script:',script,'with args:',args)
    if '.py' in script:
        cmd = 'python'
        cmd_line = [cmd,script]
    else:
        cmd_line = [script]

    for arg in args:
        cmd_line = cmd_line + [str(arg)]
    print(cmd_line)

    f_log = open(log_file+'_out.txt', 'a')

    try:
        process_exit_info = subprocess.run(cmd_line, stdout=f_log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        f_log.close()
        exit_code = e.returncode
        print('ERROR (',exit_code,'):',os.getppid(),'/',os.getpid(),script,args)
        sys.exit(exit_code)

    f_log.close()

    print(os.getppid(),'/',os.getpid(),script,'returned exit info:',process_exit_info)
    sys.exit(process_exit_info.returncode)

class ProcessManager:
    def __init__(self, max_processes, next_process_delay=10, print_status_delay=15, log_path='./00_Logs'):
        self.name = ''
        self.max_processes = max_processes
        self.processes = {}
        self.start_timers = {}
        self.completed = {}
        self.duration = {}
        self.log_path = log_path
        pm_safe_mkdir(self.log_path)
        self.next_process_delay = next_process_delay
        self.print_status_delay = print_status_delay

    def add(self, name, script, args, output_paths = None):
        if not force_run:
            paths_exist = True
            if output_paths is None:
                paths_exist = False
            else:
                for output_path in output_paths:
                    if (not file_exists(output_path)):
                        paths_exist = False
                        break
            if paths_exist:
                print('OUTPUT EXISTS:',script,output_paths,'skipping...')
                return

        log_file = self.log_path + '/log.'+name
        p = Process(target=pm_run_script, args=(script,args,log_file))
        p.name = name
        self.processes[p.name] = p
        return

    def start(self, name, max_process_limit = None):
        self.name = name
        if max_process_limit is None:
            max_process_limit = self.max_processes
        alive_count = 1
        while(alive_count > 0):
            alive_count = 0
            print('') #Empty line
            for p in self.processes.values():
                if p.pid is not None:
                    if p.is_alive():
                        alive_count += 1
                        self.duration[p.name] = timedelta(seconds=time.perf_counter() - self.start_timers[p.name])
                        print('RUNNING:',p.name,'pid:',p.pid,'is still alive','duration:',self.duration[p.name])
                    else: #No longer alive
                        if self.completed[p.name] is False:
                            self.completed[p.name] = True
                            self.duration[p.name] = timedelta(seconds=time.perf_counter() - self.start_timers[p.name])
                            print('COMPLETED:',p.name,'pid:',p.pid,'exitcode:',p.exitcode,'duration:',self.duration[p.name])
                        else: #Already marked completed
                            print('COMPLETED:',p.name,'pid:',p.pid,'exitcode:',p.exitcode,'duration:',self.duration[p.name])
                else: #No p.pid, still waiting to run
                    print('WAITING:',p.name)
            if (alive_count <= max_process_limit):
                for p in self.processes.values():
                    if p.pid is None:
                        #Sleep for 60 seconds before starting if others have already started
                        if (alive_count > 0):
                            print('Waiting for '+str(self.next_process_delay)+' secs before starting next process...')
                            time.sleep(self.next_process_delay)
                        p.start()
                        print('started processid:',p.pid,'for',p.name)
                        self.start_timers[p.name] = time.perf_counter()
                        self.completed[p.name] = False
                        self.duration[p.name] = timedelta(seconds=time.perf_counter() - self.start_timers[p.name])
                        alive_count += 1
                    if (alive_count >= max_process_limit):
                        break

            if alive_count > 0:
                print(self.name,'alive_count:',alive_count)
                #NOTE: the psutil.cpu_percent(x) call blocks for x seconds
                print('\n\nCPU: ' + str(psutil.cpu_percent(self.print_status_delay)) + '% | RAM: ' + str(psutil.virtual_memory()[2]) + '% | RAM Total Usage: ' + str(psutil.virtual_memory()[0] - psutil.virtual_memory()[1]))

        has_error = False
        error_code = 0
        for p in self.processes.values():
            if (p.exitcode != 0):
                print('ERROR:',p.name,'pid:',p.pid,'exitcode:',p.exitcode)
                has_error = True
                error_code = p.exitcode
            else:
                print(p.name,'pid:',p.pid,'COMPLETED!')
        if has_error:
            print(self.name,'has encountered an error exitcode:',error_code)
            return error_code

        self.processes = {}

        return 0
