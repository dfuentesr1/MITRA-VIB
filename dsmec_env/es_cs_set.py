import random
from config import GlobalConfig
from dsmec_env.es_cs import Station
from dsmec_env.user_device import UserDevice


class StationSet:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.es_cs_set_config = global_config.es_cs_set_config

        self.task_data_size_list = self.es_cs_set_config.task_config.task_data_size_now
        self.task_tolerance_delay_list = self.es_cs_set_config.task_config.task_tolerance_delay_list
        self.task_data_index_list = self.es_cs_set_config.task_config.task_data_index_list

        self.es_cs_list = []
        self.all_user_device_list = []

        self.es_cs_num = self.es_cs_set_config.es_cs_num
        self.user_device_num = self.es_cs_set_config.user_device_num
        self.es_cs0 = Station(0, global_config)
        self.es_cs1 = Station(1, global_config)
        self.es_cs2 = Station(2, global_config)

        self.es_cs0.computing_ability_now = \
            self.es_cs_set_config.es_cs_config.es_cs_computing_ability_list[0]
        self.es_cs1.computing_ability_now = \
            self.es_cs_set_config.es_cs_config.es_cs_computing_ability_list[1]
        self.user_device0 = UserDevice(0, global_config)
        self.user_device1 = UserDevice(1, global_config)
        self.user_device2 = UserDevice(2, global_config)
        self.user_device3 = UserDevice(3, global_config)
        self.user_device4 = UserDevice(4, global_config)
        self.user_device5 = UserDevice(5, global_config)
        self.user_device6 = UserDevice(6, global_config)

        self.es_cs_list = [self.es_cs0, self.es_cs1, self.es_cs2]
        self.all_user_device_list = [self.user_device0, self.user_device1, self.user_device2,
                                       self.user_device3, self.user_device4, self.user_device5,
                                       self.user_device6]

        assert len(self.es_cs_list) == self.es_cs_num
        assert len(self.all_user_device_list) == self.user_device_num

    def update_state(self):
        pass

    def shuffle_task_size_list(self):
        assert len(self.task_data_size_list) == len(self.task_tolerance_delay_list)
        shuffled_list = random.sample(self.task_data_index_list, len(self.task_data_index_list))
        self.task_data_size_list = self.es_cs_set_config.task_config.task_data_size_now = [
            self.task_data_size_list[self.task_data_index_list.index(i)] for i in shuffled_list]
        self.task_tolerance_delay_list = self.es_cs_set_config.task_config.task_tolerance_delay_list = [
            self.task_tolerance_delay_list[self.task_data_index_list.index(i)] for i in shuffled_list]
        self.task_data_index_list = self.es_cs_set_config.task_config.task_data_index_list = shuffled_list

    def update_all_user_device_message(self):
        for user_device_id, user_device in enumerate(self.all_user_device_list):
            user_device.create_task(user_device_id)

    def get_state_per_user_device(self):
        pass

    def draw_image(self):
        pass