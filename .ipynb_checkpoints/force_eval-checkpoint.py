# System imports
from time import sleep
from pathlib import Path
from collections.abc import Iterable   # import directly from collections for Python < 3.3

from datastream_formats import *
import numpy as np

import rclpy
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_services_default

## ML related library
from torch.utils.data import DataLoader
from torch import nn
import torch
import pickle
import joblib
data_config = {'ext_force':JointStateData('/external_ft_sensor'),
               'pressure_command':IOStatesData('/io_and_status_controller/io_states'),
               'finger_sensors1':ArrayData('/ae1_values', Float32MultiArray),
               'finger_sensors2':ArrayData('/ae2_values', Float32MultiArray)
               }
#               'tcp':TCPData("/tcp")}

class GRUModel_V2(nn.Module):
    def __init__(self, input_dim : int, hidden_dim:int, layer_dim:int, output_dim:int, dropout_prob:float,device):
        super(GRUModel_V2, self).__init__()
        self.hidden_size = hidden_dim

        # Define the RNN layer
        self.rnn = nn.GRU(input_dim, hidden_dim,num_layers=layer_dim, batch_first=True,dropout=dropout_prob)
        
        self.act_F=nn.Tanh()
        # Define the fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        # x: input tensor of shape (batch_size, sequence_length, input_size)
        # h0: initial hidden state (optional)

        # RNN layer
        out, hn = self.rnn(x, h0)
        
        out = self.act_F(out)
        # Select the last time step's output
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out




class ForceEstimator(RosDataHandler):
    """Estimate the external forces from tactile sensors."""
    def __init__(self, data_streams, seq_len=1):
        super().__init__(data_streams)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cal=True
        self.cal_iter=0
        self.model=GRUModel_V2(
            input_dim=28,
            hidden_dim=5,
            layer_dim=2,
            output_dim=3,
            dropout_prob=0,
            device=self.device)
        self.model.load_state_dict(torch.load("./online_regression/st.pth",map_location=torch.device(self.device)))


        
        self.start_datastreams(ros_version = "ros2")

        self.seq_len = seq_len
        self.vecs = []
        self.cal_val=[]
        self.x_scaler=joblib.load('./online_regression/X_st.save')
        self.y_scaler=joblib.load('./online_regression/Y_st.save')

        print("Waiting for messages to arrive")
        while not self.all_streams_recieved():
            rclpy.spin_once(self.node)
        print("All topics recieved")

        self.msg_callback_sub = self.subscriber_factory("/external_ft_sensor",
                                                        JointState,
                                                        self.msg_callback)

        self.est_pub = self.node.create_publisher(Float32MultiArray,
                                                  '/estimated_force',
                                                  qos_profile=qos_profile_services_default)


    def msg_callback(self, msg):
        data_dict = self.get_data()

        # TODO: Format into a vector
       # vec = np.array(list(data_dict['ext_force'])[:3] + list(data_dict['finger_sensors']))
        Output=torch.FloatTensor(np.array(list(data_dict['ext_force'][:3])))
        I_Pre=torch.FloatTensor(np.array(list(data_dict['pressure_command'])))
        if self.cal==True:
            self.cal_iter=self.cal_iter+1
            self.cal_val=np.array(list(data_dict['finger_sensors1'])+list(data_dict['finger_sensors2']))
            if self.cal_iter >100:
                
                self.cal=False

        Sensor=torch.FloatTensor(np.array(list(data_dict['finger_sensors1'])+list(data_dict['finger_sensors2']))-self.cal_val)
        input_f=torch.unsqueeze(torch.cat((Sensor,I_Pre),0),0)
        vec=self.x_scaler.transform(input_f)
        self.vecs.append(vec)
        if len(self.vecs) > self.seq_len:
            self.vecs.pop(0)
        vec_seq = torch.FloatTensor(np.vstack(self.vecs))
       
        #print(vec_seq)
        #print(f'size: {vec_seq.shape}')
       
        
        #print(vec_seq.shape)

        # TODO: Add your network evaluation here!
        #print(msg.effort)
        self.model.eval()
        with torch.no_grad():
           prediction=self.model(torch.unsqueeze(vec_seq,0))
           est_val=self.y_scaler.inverse_transform(prediction)
        #print(est_val)
        if self.cal_iter >100:
            msg_est_force = Float32MultiArray()
            msg_est_force.data = est_val.flatten().tolist()
            self.est_pub.publish(msg_est_force)
        else :
            print(self.cal)

        

        


def start_node():
    force_estimator = ForceEstimator(data_config, seq_len = 2)
    rclpy.spin(force_estimator.node)

if __name__ == '__main__':
    start_node()
