import multiprocessing
import time

def worker(s, i):
    s.acquire()
    print(multiprocessing.current_process().name + " acquire")
    time.sleep(i)
    print(multiprocessing.current_process().name + " release")
    s.release()

if __name__ == '__main__':
    s = multiprocessing.Semaphore(2)  # 最多允许2个进程进入，否则阻塞
    p = multiprocessing.Process(target=worker, args=(s, 2))
    p.start()
