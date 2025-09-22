from dsmec_env.queue import TaskQueue
from config import GlobalConfig


class Station:
    def __init__(self, es_cs_id, global_config: GlobalConfig) -> None:
        self.es_cs_config = global_config.es_cs_set_config.es_cs_config
        self.es_cs_id = es_cs_id

        self.computing_ability_max = self.es_cs_config.es_cs_computing_ability_max
        self.computing_ability_now = self.computing_ability_max

        self.global_config = global_config
        self.priority_task_list = []
        self.task_queue = TaskQueue(self, global_config)

        self.task_queue_current_data_size = 0
        self.task_queue_size_max = self.es_cs_config.task_queue_size_max


