import psutil
import pynvml
import time
import csv
import os
from datetime import datetime

# Initialize Sensors
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

# Dataset Configuration
csv_file = "rtx4050_telemetry_master.csv"
file_exists = os.path.isfile(csv_file)
fields = ["timestamp", "cpu_load_percent", "ram_usage_percent", "gpu_load_percent", "gpu_temp_c", "gpu_power_w"]

print("Logging started. Boot up your game!")
print("Press Ctrl+C in this terminal window to stop logging and save.")

try:
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        
        # Infinite loop until you press Ctrl+C in the terminal
        while True:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_load_percent": psutil.cpu_percent(interval=None),
                    "ram_usage_percent": psutil.virtual_memory().percent,
                    "gpu_load_percent": gpu_util.gpu,
                    "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU),
                    "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
                }
                
                writer.writerow(row)
                f.flush() 
                time.sleep(1)
                
            except Exception as sensor_error:
                # If a sensor fails, print the exact error instead of hiding it
                print(f"Sensor read error: {sensor_error}")
                time.sleep(1)

except KeyboardInterrupt:
    # This safely catches the Ctrl+C command
    print("\nLogging stopped successfully by user.")
finally:
    pynvml.nvmlShutdown()