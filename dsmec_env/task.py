from config import GlobalConfig
import numpy as np


class Task:
    def __init__(self, user_device_id, global_config: GlobalConfig):
        self.task_config = global_config.es_cs_set_config.task_config
        self.task_from_user_device_id = user_device_id

        self.task_data_size = np.random.normal(self.task_config.task_data_size_now[user_device_id],
                                               self.task_config.task_date_size_std).item()
        self.task_data_size_max = self.task_config.task_data_size_max
        self.task_current_process_time_in_queue = 0

        self.task_local_finish_time = 0
        self.task_offload_finish_time = 0
        self.task_tolerance_delay = self.task_config.task_tolerance_delay_list[user_device_id]
        self.task_tolerance_delay_max = self.task_config.task_tolerance_delay_max

        self.step_count_begin = -1
        self.task_switch_time_list_on_es_cs = self.task_config.task_switch_time_matrix_on_es_cs[user_device_id]

    def get_task_info_list(self):
        return [self.task_data_size, self.task_tolerance_delay]
