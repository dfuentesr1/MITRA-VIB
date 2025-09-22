from config import GlobalConfig
from dsmec_env.task import Task
from dsmec_env.queue import TaskQueue


class UserDevice:
    def __init__(self, user_device_id, global_config: GlobalConfig):
        self.user_device_id = user_device_id
        self.belong_es_cs = None
        self.global_config = global_config
        self.user_device_config = global_config.es_cs_set_config.user_device_config
        self.last_es_cs_offload_choice = -1

        self.transmitting_time_to_all_es_cs = \
            self.user_device_config.transmitting_time_to_all_es_cs_array[self.user_device_id]
        self.transmitting_time_to_es_cs_max = self.user_device_config.transmitting_time_to_es_cs_max

        self.computing_ability_max = self.user_device_config.user_device_ability_max
        self.computing_ability_now = self.user_device_config.user_device_ability

        self.task = None
        self.task_queue = TaskQueue(self, global_config)

        self.task_queue_current_data_size = 0
        self.task_queue_size_max = self.user_device_config.task_queue_size_max

    def create_task(self, user_device_id):
        self.task = Task(user_device_id, self.global_config)

    def update_task(self, user_device_id):
        self.task = Task(user_device_id, self.global_config)
