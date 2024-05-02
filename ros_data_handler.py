from time import sleep
from rosbags.highlevel import AnyReader


# Comment: time stamps may not be trustable between different sources;
#          or even the rosbag recorder which takes them.

class RosDataHandler():
    '''
    Class for consistent formatting, pre-processing, and sync of ROS
    messages both online (subscribers) and offline (rosbags).

    A single configuration file determines:
    - How a single topic is formatted
    - How the topic messages should be pre-processed (linear transform)
    - How a collection of topics should be synchronized by specifying which
      topic acts as the clock

    This config file is used to generate a rosbag reader and subscriber dict
    with the same formatting. 
    '''
    def __init__(self, data_streams):    
        self.data_streams = data_streams
        self.last_msg = {k:None for k in self.data_streams.keys()}
        self.dataset = {k:[] for k in self.data_streams.keys()}
        self.vectorset = []
        self.build_callbacks()

    def start_datastreams(self,
                          ros_version = 'ros1',
                          callback_fn = None,
                          callback_data_stream = None):
        if ros_version == 'ros1':
            self.init_ros1()
        elif ros_version == 'ros2':
            self.init_ros2()
        else:
            raise Exception("ros_version should be ros1 or ros2")

        self.build_subscribers()
        if callback_fn and callback_data_stream:
            self.subscribers['user_callback'] = self.subscriber_factory(data_stream.topic_name,
                                                                        data_stream.msg_type,
                                                                        callback_fn)
            
    def init_ros1(self):
        import rospy
        rospy.init_node('data_handler')
        self.subscriber_factory = rospy.Subscriber
        
    def init_ros2(self):
        import rclpy
        from rclpy.node import Node
        rclpy.init(args=None)
        self.node = Node('data_handler')
        rearranged_fn = lambda name, typ, fn : self.node.create_subscription(typ, name, fn, 1) 
        self.subscriber_factory = rearranged_fn
        #self.sleep_fn = lambda time: self.node.create_rate(time).sleep()
            
    # Build ROS subscribers for each topic in data
    def build_subscribers(self):
        self.subscribers = {}
        for stream_name, data_stream in self.data_streams.items():
            print(f'Adding subscriber to {data_stream.topic_name}')
            self.subscribers[stream_name] = self.subscriber_factory(data_stream.topic_name,
                                                                    data_stream.msg_type,
                                                                    self.callbacks[stream_name])
        
    # Builds callback functions for all datastreams,
    # binding the map formatter to update the local variable
    def build_callbacks(self):
        self.callbacks = {}
        for stream_name, data_stream in self.data_streams.items():
            #print(f'Building callback for {stream_name}')
            self.callbacks[stream_name] = self.callback_factory(stream_name, data_stream)

    def callback_factory(self, stream_name, data_stream):
        def topic_callback(msg):
            self.last_msg[stream_name] = data_stream.map_msg(msg)
        return topic_callback
    
    def all_streams_recieved(self):
        return all(val is not None for val in self.last_msg.values())

    def wait_all_streams_recieved(self):
        print("Waiting on data over ROS")
        while not self.all_streams_recieved():
            sleep(0.001)
        print("All datastreams recieved!")

    def load_bags(self, bag_paths, sync_stream = None):
        '''
        IN
         bag_paths: list of paths to the rosbags
         topic: topic to load from all of the bags
        OUT
         topic_msgs: dict formatted according to topic.append_msg
        '''
        with AnyReader(bag_paths) as bagreader:
            topics = [v.topic_name for v in self.data_streams.values()]
            connections = [x for x in bagreader.connections if x.topic in topics]
            for conn, timestamp, rawdata in bagreader.messages(connections = connections):
                msg = bagreader.deserialize(rawdata, conn.msgtype)
                # Find all datastreams derived from this message topic
                streams = [k for k,v in self.data_streams.items() if v.topic_name == conn.topic]
                for stream_name in streams:
                    self.callbacks[stream_name](msg)

                # If sync stream has been updated, append to the dataset and vectorset
                if sync_stream and self.all_streams_recieved() and (sync_stream in streams):
                    for k in self.data_streams.keys():
                        self.dataset[k].append(self.last_msg[k])
                    self.vectorset.append(self.get_vector())
                    
    def get_data(self):
        return self.last_msg
        
    # Returns synchronized data across all subscribers
    def get_vector(self):
        assert self.all_streams_recieved(), 'Easy there! Not all topics recieved yet'
        return [elem for data in self.last_msg.values() for elem in data]
        
class DataStream():
    '''
    A wrapper class for a specific datastream which specifies:
     how to format from the ros message to a python data structure,
     ros topic associated with it
    '''
    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.msg_type = None
    
    def map_msg(self, msg, data = {}):
        '''
        Read msg, apply transform, and append it to the data iterable, if given
        '''
        return data
    
