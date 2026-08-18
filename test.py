import torch
import multiprocessing

def worker():
    print("worker ok")

if __name__ == "__main__":
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()