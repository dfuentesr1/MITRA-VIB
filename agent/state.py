import numpy as np

from dsmec_env.es_cs_set import StationSet
from dsmec_env.user_device import UserDevice


class State:
    def __init__(self, user_device: UserDevice, es_cs_set: StationSet):
        self.user_device = user_device
        self.user_device_id = user_device.user_device_id

        self.es_cs_list = es_cs_set.es_cs_list
        self.user_device_computing_ability = user_device.computing_ability_now  # for BS_com_ability_list
        self.user_device_list = es_cs_set.all_user_device_list  # for other task size and mask
        self.task_data_size = user_device.task.task_data_size
        self.task_tolerance_delay = user_device.task.task_tolerance_delay
        self.task_data_index_list = es_cs_set.es_cs_set_config.task_config.task_data_index_list

        self.user_device_task_queue_current_data_size = user_device.task_queue_current_data_size
        self.user_device_task_queue_size_max = user_device.task_queue_size_max
        self.es_cs_task_current_sum_process_time_list = []
        self.es_cs_task_queue_size_max_list = [es_cs.task_queue_size_max for es_cs
                                                      in self.es_cs_list]

        self.es_cs_set_computing_ability_list = self.get_es_cs_set_computing_ability_list(
            self.es_cs_list)

        self.all_task_size_list, self.task_size_mask_list = self.get_other_task_size(self.user_device_list,
                                                                                     self.user_device_id)
        self.transmitting_time_to_all_es_cs_list = self.user_device.transmitting_time_to_all_es_cs.tolist()
        self.es_cs_task_current_sum_process_time_list = self.get_es_cs_task_current_sum_process_time_list(
            self.es_cs_list)
        self.last_es_cs_offload_choice = user_device.last_es_cs_offload_choice

    def get_es_cs_set_computing_ability_list(self, es_cs_list):
        es_cs_set_computing_ability_list = []
        for es_cs in es_cs_list:
            es_cs_set_computing_ability_list.append(es_cs.computing_ability_now)
        return es_cs_set_computing_ability_list

    def get_other_task_size(self, user_device_list, user_device_id):
        all_task_size_list = []
        task_size_mask_list = []
        mask_item = 0
        for idx, user_device in enumerate(user_device_list):
            if self.task_data_index_list[user_device_id] == idx:
                mask_item = 1
            else:
                mask_item = 0
            all_task_size_list.append(user_device_list[self.task_data_index_list.index(idx)].task.task_data_size)
            task_size_mask_list.append(mask_item)
        return all_task_size_list, task_size_mask_list

    def get_es_cs_task_current_sum_process_time_list(self, es_cs_list):
        es_cs_task_current_sum_process_time_list = []
        for es_cs in es_cs_list:
            task_current_sum_process_time = es_cs.task_queue.get_task_current_sum_process_time()
            es_cs_task_current_sum_process_time_list.append(task_current_sum_process_time)
        return es_cs_task_current_sum_process_time_list

    def get_state_list(self):
        state_list = []
        es_cs_set_computing_ability_list = self.get_es_cs_set_computing_ability_list(
            self.es_cs_list)
        state_list.extend(es_cs_set_computing_ability_list)
        state_list.append(self.user_device_computing_ability)
        state_list.append(self.task_data_size)
        state_list.append(self.task_tolerance_delay)
        self.all_task_size_list, self.task_size_mask_list = self.get_other_task_size(self.user_device_list,
                                                                                     self.user_device_id)
        self.transmitting_time_to_all_es_cs_list = self.user_device.transmitting_time_to_all_es_cs.tolist()
        self.es_cs_task_current_sum_process_time_list = self.get_es_cs_task_current_sum_process_time_list(
            self.es_cs_list)

        state_list += self.all_task_size_list + self.task_size_mask_list + self.transmitting_time_to_all_es_cs_list + self.es_cs_task_current_sum_process_time_list
        self.last_es_cs_offload_choice = self.user_device.last_es_cs_offload_choice
        state_list.append(self.last_es_cs_offload_choice)

        return state_list

    def get_normalized_state_array(self):
        state_array = self.get_state_array()
        normalized_state_array = np.zeros_like(state_array)

        es_cs_computing_ability_max = self.es_cs_list[0].computing_ability_max
        normalized_state_array[:2] = state_array[:2] / es_cs_computing_ability_max

        user_device_computing_ability_max = self.user_device_list[0].computing_ability_max
        normalized_state_array[2] = state_array[2] / user_device_computing_ability_max

        task_data_size_max = self.user_device.task.task_data_size_max
        normalized_state_array[3] = state_array[3] / task_data_size_max
        normalized_state_array[4] = state_array[4] / self.user_device.task.task_tolerance_delay_max
        normalized_state_array[5:8] = state_array[5:8] / task_data_size_max
        normalized_state_array[8:11] = state_array[8:11]

        normalized_state_array[11:13] = state_array[11:13] / self.user_device_list[0].transmitting_time_to_es_cs_max

        normalized_state_array[-3:] = state_array[-3:]

        return normalized_state_array

    def get_state_array(self):
        import numpy as np
        return np.array(self.get_state_list())
