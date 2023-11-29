import random
import numpy as np
from sleepecg import load_classifier, stage , SleepRecord , extract_features , plot_hypnogram
# importing matplotlib for local uses
import matplotlib.pyplot as plt
import random
data = [

    {'TimeStamp': '1690409900', 'HeartRate': '55'},
    {'TimeStamp': '1690410000', 'HeartRate': '60'},
    {'TimeStamp': '1690410120', 'HeartRate': '65'},
    {'TimeStamp': '1690410300', 'HeartRate': '62'},
    {'TimeStamp': '1690410600', 'HeartRate': '59'},
    {'TimeStamp': '1690410900', 'HeartRate': '58'},
    {'TimeStamp': '1690411200', 'HeartRate': '61'},
    {'TimeStamp': '1690411320', 'HeartRate': '56'},
    {'TimeStamp': '1690411380', 'HeartRate': '52'},
    {'TimeStamp': '1690411440', 'HeartRate': '50'},
    {'TimeStamp': '1690411500', 'HeartRate': '56'},
    {'TimeStamp': '1690411560', 'HeartRate': '58'},
    {'TimeStamp': '1690411620', 'HeartRate': '60'},
    {'TimeStamp': '1690411680', 'HeartRate': '65'},
    {'TimeStamp': '1690412740', 'HeartRate': '60'},
    {'TimeStamp': '1690412800', 'HeartRate': '62'},
    {'TimeStamp': '1690413800', 'HeartRate': '62'},
    {'TimeStamp': '1690514000', 'HeartRate': '62'}
]

clf = load_classifier("wrn-gru-mesa-weighted", "SleepECG")
def sleepanalyse(data , min_treshhold = 1500 , sleep_stage_duration = 60) : 
    # Convert to NumPy arrays
    timestamps = [int(d['TimeStamp']) for d in data]
    heart_rates = [int(d['HeartRate']) for d in data]


    # Define a function to generate R-peak times on-the-fly to reduce cpu usage
    # this function generates r peaks based on timeinterval between two timestamp and heart rate

    def generate_synthetic_r_peaks(timestamps, heart_rates):
        peak_count = 0
        current_time = 0  # in unix seconds, relative to the start of recording
        max_peaks = 0 
        for i in range(len(timestamps) - 1):
            time_interval = timestamps[i + 1] - timestamps[i]
            n_peaks = int(time_interval / (60.0 / heart_rates[i]))
            max_peaks += n_peaks

            for j in range(n_peaks):
                if peak_count >= max_peaks:
                    return
                r_peak_interval = (60.0 / heart_rates[i]) + random.uniform(0.001, 0.01)  # Add a small random
                current_time += r_peak_interval
                yield current_time  # Yield the R-peak time as needed
                peak_count += 1



    # Identify long intervals
    long_interval_indices = []
    for i in range(len(timestamps) - 1):
        time_interval = timestamps[i + 1] - timestamps[i]
        if time_interval > 1800:  # 30 minutes
            long_interval_indices.append(i)
            
 

    combined_stages_pred = []
    aggregated_heartbeat_times = []
    start_idx = 0
    for end_idx in long_interval_indices:
        heartbeat_times_list = list(generate_synthetic_r_peaks(
            timestamps[start_idx:end_idx + 1], heart_rates[start_idx:end_idx + 1]
        ))
        aggregated_heartbeat_times.extend(heartbeat_times_list)
            # Check if heartbeat_times_list has enough elements 
        if len(heartbeat_times_list) > min_treshhold: 
            rec = SleepRecord(
                sleep_stage_duration=sleep_stage_duration,
                heartbeat_times=heartbeat_times_list
            )
            stages_pred = stage(clf, rec, return_mode="prob")
            print(f"this is stages pred ,,,,,,,,,{stages_pred} : ")
            combined_stages_pred.extend(stages_pred)
        else:
            print(f"Skipping segment starting at start index ({start_idx}) of data  because of not enough data.")
            time_interval = timestamps[end_idx + 1] - timestamps[end_idx]
            num_placeholder_blocks = int(time_interval / sleep_stage_duration)
            placeholder_array = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            combined_stages_pred.extend([placeholder_array] * num_placeholder_blocks)
        
        # Add placeholder for the long interval
        #the place holder is [0,0,0,0]
        time_interval = timestamps[end_idx + 1] - timestamps[end_idx]
        num_placeholder_blocks = int(time_interval / sleep_stage_duration)
        placeholder_array = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        combined_stages_pred.extend([placeholder_array] * num_placeholder_blocks)
        start_idx = end_idx + 1



    # Handle the last segment if needed
    if start_idx < len(timestamps) - 1:
        heartbeat_times_list = list(generate_synthetic_r_peaks(
            timestamps[start_idx:], heart_rates[start_idx:]
    ))
        if len(heartbeat_times_list) > min_treshhold:
            rec = SleepRecord(
                sleep_stage_duration=sleep_stage_duration,
                heartbeat_times=heartbeat_times_list
            )
            stages_pred = stage(clf, rec, return_mode="prob")
            combined_stages_pred.extend(stages_pred)
        else:
            print(f"Skipping the last segment because of not enough data.")
            time_interval = timestamps[-1] - timestamps[start_idx]
            num_placeholder_blocks = int(time_interval / sleep_stage_duration)
            placeholder_array = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            combined_stages_pred.extend([placeholder_array] * num_placeholder_blocks)


    heartbeat_times_list = list(generate_synthetic_r_peaks(
        timestamps[start_idx:], heart_rates[start_idx:]
    ))
    aggregated_heartbeat_times.extend(heartbeat_times_list)

    # Create a SleepRecord object with aggregated heartbeat times
    final_rec = SleepRecord(
        sleep_stage_duration=sleep_stage_duration,
        heartbeat_times=aggregated_heartbeat_times
    )

    # Convert to a NumPy array for easier modiy later if needed
    combined_stages_pred = np.array(combined_stages_pred)
    # for using in flask
    stages_mode=clf.stages_mode

    return final_rec , combined_stages_pred , stages_mode , long_interval_indices

final_rec , combined_stages_pred , stages_mode ,long_interval_indices = sleepanalyse(data)


# in case of ploting with matplotlib for local uses 

plot_hypnogram(
    final_rec,
    combined_stages_pred,
    stages_mode=stages_mode
)
plt.show()
