import tensorflow as tf
import numpy as np
import json

#%
PATHS = [
    {
        "tfrecords" : "dataset/dichasus-cf02.tfrecords",
        "offsets" : "dataset/reftx-offsets-dichasus-cf02.json"
    },
    {
        "tfrecords" : "dataset/dichasus-cf03.tfrecords",
        "offsets" : "dataset/reftx-offsets-dichasus-cf03.json"
    },
    {
        "tfrecords" : "dataset/dichasus-cf04.tfrecords",
        "offsets" : "dataset/reftx-offsets-dichasus-cf04.json"
    }
]

spec = None

antenna_assignments = []
antenna_count = 0

with open("dataset/spec.json") as specfile:
    spec = json.load(specfile)
    AP_pos = np.zeros((4,3))
    for idx, antenna in enumerate(spec["antennas"]):
        antenna_count = antenna_count + sum([len(row) for row in antenna["assignments"]])
        antenna_assignments.append(antenna["assignments"])
        AP_pos[idx] = antenna["location"] # sue

def load_calibrate(path, offset_path):
    offsets = None
    with open(offset_path, "r") as offsetfile:
        offsets = json.load(offsetfile)

    def record_parse_function(proto):
        record = tf.io.parse_single_example(
            proto,
            {
                "csi": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "time": tf.io.FixedLenFeature([], tf.float32, default_value = 0),
            },
        )

        csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (antenna_count, 1024, 2))
        csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
        csi = tf.signal.fftshift(csi, axes=1)

        position = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))
        time = tf.ensure_shape(record["time"], ())

        return csi, position, time

    def apply_calibration(csi, pos, time):
        sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
        cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))

        return csi, pos, time

    def order_by_antenna_assignments(csi, pos, time):
        csi = tf.stack([[tf.gather(csi, antenna_indices) for antenna_indices in array] for array in antenna_assignments])
        return csi, pos, time

    def fdomain_csi_to_mean_delayspread(csi):
        # csi comes in as shape (arrays, rows, columns, subcarriers)
        # --> we want (arrays, antennas = rows * columns, subcarriers), reshape!
        csi = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes = -1)), axes=-1)[:,:,:,512-8:512+40]
        csi_by_array = tf.reshape(csi, (tf.shape(csi)[0], 8, tf.shape(csi)[3]))
        powers = tf.square(tf.abs(csi_by_array))
        timestamps = tf.range(tf.shape(csi_by_array)[2], dtype = np.float32) / spec["bandwidth"]
        time_weighted_powers = tf.einsum("amt,t->amt", powers, timestamps)
        mean_delays = tf.divide(tf.reduce_sum(time_weighted_powers, axis = 2), tf.reduce_sum(powers, axis = 2))
        squared_delays = tf.square(timestamps[tf.newaxis, tf.newaxis, :] - mean_delays[:, :, tf.newaxis])
        delayspreads = tf.sqrt(tf.divide(tf.reduce_sum(tf.multiply(squared_delays, powers), axis = 2), tf.reduce_sum(powers, axis = 2)))
    
        return delayspreads
    
    def is_broken_datapoint(csi, pos, time):
        # A bug in the DICHASUS postprocessing script introduced a small number of erroneous data in the
        # current version of the dc0x dataset (will be fixed in the future by re-running the postprocessing
        # script). For now, we just remove these datapoints, which can be identified by a very high delay spread
        # (if examined closely, these datapoints are totally nonsensical).
        return tf.math.reduce_all(fdomain_csi_to_mean_delayspread(csi) < 2e-7)
    
    dset = tf.data.TFRecordDataset(path)
    dset = dset.map(record_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
    dset = dset.map(apply_calibration, num_parallel_calls = tf.data.AUTOTUNE)
    dset = dset.map(order_by_antenna_assignments, num_parallel_calls = tf.data.AUTOTUNE)
    dset = dset.filter(is_broken_datapoint)

    return dset

full_dataset = load_calibrate(PATHS[0]["tfrecords"], PATHS[0]["offsets"])

for path in PATHS[1:]:
    full_dataset = full_dataset.concatenate(load_calibrate(path["tfrecords"], path["offsets"]))

# Decimate training dataset: Use only every 4th datapoint (to reduce number of points)
training_set = full_dataset.enumerate().filter(lambda idx, value : (idx % 5 == 0))
training_set = training_set.map(lambda idx, value : value)

# Use different datapoints for test set (shift by 2)
test_set = full_dataset.enumerate().filter(lambda idx, value : ((idx + 2) % 5 == 0))
test_set = test_set.map(lambda idx, value : value)

print('here')

def csi_time_domain(csi, pos, time):
    csi = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes = -1)), axes = -1)
    return csi, pos, time

def cut_out_taps(tap_start, tap_stop):
    def cut_out_taps_func(csi, pos, time):
        return csi[:,:,:,tap_start:tap_stop], pos, time

    return cut_out_taps_func

training_set = training_set.map(csi_time_domain, num_parallel_calls = tf.data.AUTOTUNE)
training_set = training_set.map(cut_out_taps(507, 520), num_parallel_calls = tf.data.AUTOTUNE)

test_set = test_set.map(csi_time_domain, num_parallel_calls = tf.data.AUTOTUNE)
test_set = test_set.map(cut_out_taps(507, 520), num_parallel_calls = tf.data.AUTOTUNE)
#%%
groundtruth_positions = []
csi_time_domain = []
timestamps = []

for csi, pos, time in training_set.prefetch(tf.data.AUTOTUNE).batch(1000):
    csi_time_domain.append(tf.reshape(csi, (tf.shape(csi)[0], tf.shape(csi)[1], 8, tf.shape(csi)[4])).numpy())
    groundtruth_positions.append(pos.numpy())
    timestamps.append(time.numpy())

#%
csi_time_domain = np.concatenate(csi_time_domain)
groundtruth_positions = np.concatenate(groundtruth_positions)
times = np.concatenate(timestamps)

#%
ii = np.argsort(times)
timestamps = times[ii]
H = np.reshape(csi_time_domain[ii], (csi_time_domain.shape[0], 32, -1))
UE_pos = groundtruth_positions[ii]

print(H.shape)
print(timestamps.shape)
print(UE_pos.shape)
print(AP_pos.shape)
print(np.sum(timestamps[1:] - timestamps[:-1] < 0))

#%
ang = np.pi / 15
rotation_matrix = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
UE_pos[:,:2] = UE_pos[:,:2] @ rotation_matrix
AP_pos[:,:2] = AP_pos[:,:2] @ rotation_matrix
np.save('UE_pos', UE_pos)
np.save('AP_pos', AP_pos)

np.save('H', H)
np.save('timestamps', timestamps)
bounding_boxes_ap = np.array([[[-5.5, -13.5], [-0.5, -1]],
                            [[-13, -7.5], [-0.5, -1]],
                            [[-5.5, -13.5], [ -0.5, -1]],
                            [[-13, -7.5], [ -0.5, -1]]]) 
np.save('bounding_boxes_APs', bounding_boxes_ap)
