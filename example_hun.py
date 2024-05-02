# System imports
from time import sleep
from pathlib import Path
from collections.abc import Iterable   # import directly from collections for Python < 3.3
import os as os

# ROS imports
from sensor_msgs.msg import JointState
from ur_msgs.msg import IOStates
from std_msgs.msg import Int16MultiArray, Float64, Float32MultiArray

from rosbags.typesys import get_types_from_msg, register_types

# Project imports
from ros_data_handler import RosDataHandler, DataStream
import numpy as np

class JointStateData(DataStream):
    def __init__(self, topic_name):
        DataStream.__init__(self, topic_name)
        self.msg_type = JointState

    def map_msg(self, msg):
        return msg.effort

class IOStatesData(DataStream):
    def __init__(self, topic_name):
        DataStream.__init__(self, topic_name)
        self.msg_type = IOStates

    def map_msg(self, msg):
        return [msg.analog_out_states[0].state,
                msg.analog_out_states[1].state]

class ArrayData(DataStream):
    def __init__(self, topic_name, msg_type):
        DataStream.__init__(self, topic_name)
        self.msg_type = msg_type

    def map_msg(self, msg):
        data = msg.data
        return data if isinstance(data, Iterable) else [data]

def add_UR_msgs():
#    msg_text = Path('/opt/ros/galactic/share/ur_msgs/msg/IOStates.msg').read_text()
#    register_types(get_types_from_msg(msg_text, '/opt/ros/galactic/share/ur_msgs/msg/IOStates.msg'))
    typs = {}
    for root, dirnames, files in os.walk('/home/ipk410/ros2_ws/src/ur_msgs/'):
        for fname in files:
            path = Path(root, fname)
            if path.suffix == '.msg':
                name = path.relative_to(path.parents[2]).with_suffix('')
                if '/msg/' not in str(name):
                    name = name.parent / 'msg' / name.name
                #print(name)
                typs.update(get_types_from_msg(path.read_text(encoding='utf-8'), name))
                #print(typs)
        typs = dict(sorted(typs.items()))
        register_types(typs)

if  __name__ == '__main__':
    add_UR_msgs()

    data_config = {'ext_force':JointStateData('/external_ft_sensor'),
                   'pressure_command':IOStatesData('/io_and_status_controller/io_states'),
                   'finger_sensors':ArrayData('/i2c_sensors', Float32MultiArray)}
    data_handler = RosDataHandler(data_config)
    data_handler.load_bags([Path('test_26_05_23_v1')], sync_stream = 'ext_force')

    dataset = data_handler.dataset.items()
    for k,v in dataset:
        print(f'Loaded up {k} with size {np.array(v).shape}')
        print(f'  first one looks like  {v[0]}')


    vectorset = data_handler.vectorset
    print(f'Also avail as a big array of size {np.array(vectorset).shape}')
    print(f'  first one looks like {vectorset[0]}')

