import pandas as pd
import time
import threading as th
import os
import logging

logging.basicConfig(
    filename="data.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

list_datasets = ['sample.csv', 'sample2.csv', 'sample3.csv']
df = {}
threads = []
lock = th.Lock()   # ensure only one thread updates dict at a time


def read_data(filename):
    logging.info(f"Started reading {filename}")
    time.sleep(2)  # simulate delay
    try:
        data = pd.read_csv(filename, encoding="ISO-8859-1")
        with lock:
            df[filename] = data
        logging.info(f"Finished reading {filename}")
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")


# Start threads
for fname in list_datasets:
    if os.path.exists(fname):
        t = th.Thread(target=read_data, args=(fname,), name=fname)
        threads.append(t)
        t.start()
    else:
        logging.error(f"File {fname} does not exist")

# Wait for all threads to finish
for t in threads:
    t.join()

logging.info("All files processed ✅")

# Print row counts
total_rows = 0
for fname, data in df.items():
    rows = data.shape[0]   # only rows
    print(f"{fname} → {rows} rows")
    logging.info(f"{fname} has {rows} rows")
    total_rows += rows


print(f"TOTAL rows = {total_rows}")
logging.info(f"Total rows across all datasets: {total_rows}")



        