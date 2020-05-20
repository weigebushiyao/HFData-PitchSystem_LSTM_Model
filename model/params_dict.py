class ParamsDict:
    model_params={
        'model_params_v1':{
        'cap_temp_1': ['pitch_Atech_hub_temp_1',
                       'pitch_Atech_cabinet_temp_1',
                       'pitch_Atech_motor_current_1',
                       'pitch_position_1', 'wind_speed', 'rotor_speed',
                       'pitch_Atech_capacitor_temp_1'],

        'cap_temp_2': ['pitch_Atech_hub_temp_2',
                       'pitch_Atech_cabinet_temp_2',
                       'pitch_Atech_motor_current_2',
                       'pitch_position_2', 'wind_speed', 'rotor_speed',
                       'pitch_Atech_capacitor_temp_2'],

        'cap_temp_3': ['pitch_Atech_hub_temp_3',
                       'pitch_Atech_cabinet_temp_3',
                       'pitch_Atech_motor_current_3',
                       'pitch_position_3', 'wind_speed',
                       'rotor_speed',
                       'pitch_Atech_capacitor_temp_3'],

        'convt_temp_1': ['pitch_Atech_hub_temp_1',
                         'pitch_Atech_cabinet_temp_1',
                         'pitch_Atech_motor_current_1',
                         'pitch_position_1', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_converter_temp_1'],

        'convt_temp_2': ['pitch_Atech_hub_temp_2',
                         'pitch_Atech_cabinet_temp_2',
                         'pitch_Atech_motor_current_2',
                         'pitch_position_2', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_converter_temp_2'],

        'convt_temp_3': ['pitch_Atech_hub_temp_3',
                         'pitch_Atech_cabinet_temp_3',
                         'pitch_Atech_motor_current_3',
                         'pitch_position_3', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_3',
                         'pitch_Atech_motor_temp_3',
                         'pitch_Atech_converter_temp_3'],

        'motor_temp_1': ['pitch_Atech_hub_temp_1',
                         'pitch_Atech_cabinet_temp_1',
                         'pitch_Atech_motor_current_2',
                         'pitch_position_1', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_1',
                         'pitch_Atech_converter_temp_1',
                         'pitch_Atech_motor_temp_1'],

        'motor_temp_2': ['pitch_Atech_hub_temp_2',
                         'pitch_Atech_cabinet_temp_2',
                         'pitch_Atech_motor_current_2',
                         'pitch_position_2', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_2',
                         'pitch_Atech_converter_temp_2',
                         'pitch_Atech_motor_temp_2'],

        'motor_temp_3': ['pitch_Atech_hub_temp_3',
                         'pitch_Atech_cabinet_temp_3',
                         'pitch_Atech_motor_current_3',
                         'pitch_position_3', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_3',
                         'pitch_Atech_converter_temp_3',
                         'pitch_Atech_motor_temp_3'], },

    'model_params_v2': {
        'cap_temp_1': ['pitch_Atech_hub_temp_1',
                       'pitch_Atech_cabinet_temp_1',
                       'pitch_Atech_motor_current_1',
                       'pitch_position_1', 'converter_power','wind_speed', 'rotor_speed',
                       'pitch_Atech_motor_temp_1',
                       'pitch_Atech_converter_temp_1',
                       'pitch_Atech_capacitor_temp_1'],

        'cap_temp_2': ['pitch_Atech_hub_temp_2',
                       'pitch_Atech_cabinet_temp_2',
                       'pitch_Atech_motor_current_2',
                       'pitch_position_2','converter_power', 'wind_speed', 'rotor_speed',
                       'pitch_Atech_motor_temp_2',
                       'pitch_Atech_converter_temp_2',
                       'pitch_Atech_capacitor_temp_2'],

        'cap_temp_3': ['pitch_Atech_hub_temp_3',
                       'pitch_Atech_cabinet_temp_3',
                       'pitch_Atech_motor_current_3',
                       'pitch_position_3', 'converter_power','wind_speed','rotor_speed',
                       'pitch_Atech_motor_temp_3',
                       'pitch_Atech_converter_temp_3',
                       'pitch_Atech_capacitor_temp_3'],

        'convt_temp_1': ['pitch_Atech_hub_temp_1',
                         'pitch_Atech_cabinet_temp_1',
                         'pitch_Atech_motor_current_1',
                         'pitch_position_1', 'converter_power','wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_1',
                         'pitch_Atech_motor_temp_1',
                         'pitch_Atech_converter_temp_1'],

        'convt_temp_2': ['pitch_Atech_hub_temp_2',
                         'pitch_Atech_cabinet_temp_2',
                         'pitch_Atech_motor_current_2',
                         'pitch_position_2','converter_power', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_2',
                         'pitch_Atech_motor_temp_2',
                         'pitch_Atech_converter_temp_2'],

        'convt_temp_3': ['pitch_Atech_hub_temp_3',
                         'pitch_Atech_cabinet_temp_3',
                         'pitch_Atech_motor_current_3',
                         'pitch_position_3', 'converter_power', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_3',
                         'pitch_Atech_motor_temp_3',
                         'pitch_Atech_converter_temp_3'],

        'motor_temp_1': ['pitch_Atech_hub_temp_1',
                         'pitch_Atech_cabinet_temp_1',
                         'pitch_Atech_motor_current_2',
                         'pitch_position_1', 'converter_power', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_1',
                         'pitch_Atech_converter_temp_1',
                         'pitch_Atech_motor_temp_1'],

        'motor_temp_2': ['pitch_Atech_hub_temp_2',
                         'pitch_Atech_cabinet_temp_2',
                         'pitch_Atech_motor_current_2',
                         'pitch_position_2', 'converter_power', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_2',
                         'pitch_Atech_converter_temp_2',
                         'pitch_Atech_motor_temp_2'],

        'motor_temp_3': ['pitch_Atech_hub_temp_3',
                         'pitch_Atech_cabinet_temp_3',
                         'pitch_Atech_motor_current_3',
                         'pitch_position_3', 'converter_power', 'wind_speed', 'rotor_speed',
                         'pitch_Atech_capacitor_temp_3',
                         'pitch_Atech_converter_temp_3',
                         'pitch_Atech_motor_temp_3'], },
        'model_params_v3': {
            'cap_temp_1': ['pitch_Atech_hub_temp_1',
                           'pitch_Atech_cabinet_temp_1',
                           'pitch_Atech_motor_current_1',
                           'pitch_position_1', 'converter_power', 'wind_speed', 'rotor_speed',
                           'pitch_Atech_motor_temp_1',
                           'pitch_Atech_converter_temp_1',
                           'pitch_Atech_capacitor_temp_1'],

            'cap_temp_2': ['pitch_Atech_hub_temp_2',
                           'pitch_Atech_cabinet_temp_2',
                           'pitch_Atech_motor_current_2',
                           'pitch_position_2', 'converter_power', 'wind_speed', 'rotor_speed',
                           'pitch_Atech_motor_temp_2',
                           'pitch_Atech_converter_temp_2',
                           'pitch_Atech_capacitor_temp_2'],

            'cap_temp_3': ['pitch_Atech_hub_temp_3',
                           'pitch_Atech_cabinet_temp_3',
                           'pitch_Atech_motor_current_3',
                           'pitch_position_3', 'converter_power', 'wind_speed', 'rotor_speed',
                           'pitch_Atech_motor_temp_3',
                           'pitch_Atech_converter_temp_3',
                           'pitch_Atech_capacitor_temp_3'],

            'convt_temp_1': ['pitch_Atech_hub_temp_1',
                             'pitch_Atech_cabinet_temp_1',
                             'pitch_Atech_motor_current_1',
                             'pitch_position_1', 'converter_power', 'wind_speed', 'rotor_speed',
                             'pitch_Atech_capacitor_temp_1',
                             'pitch_Atech_motor_temp_1',
                             'pitch_Atech_converter_temp_1'],

            'convt_temp_2': ['pitch_Atech_hub_temp_2',
                             'pitch_Atech_cabinet_temp_2',
                             'pitch_Atech_motor_current_2',
                             'pitch_position_2', 'converter_power', 'wind_speed', 'rotor_speed',
                             'pitch_Atech_capacitor_temp_2',
                             'pitch_Atech_motor_temp_2',
                             'pitch_Atech_converter_temp_2'],

            'convt_temp_3': ['pitch_Atech_hub_temp_3',
                             'pitch_Atech_cabinet_temp_3',
                             'pitch_Atech_motor_current_3',
                             'pitch_position_3', 'converter_power', 'wind_speed', 'rotor_speed',
                             'pitch_Atech_capacitor_temp_3',
                             'pitch_Atech_motor_temp_3',
                             'pitch_Atech_converter_temp_3'],

            'motor_temp_1': ['pitch_Atech_hub_temp_1',
                             'pitch_Atech_cabinet_temp_1',
                             'pitch_Atech_motor_current_2',
                             'pitch_position_1', 'converter_power', 'wind_speed', 'rotor_speed',
                             'pitch_Atech_capacitor_temp_1',
                             'pitch_Atech_converter_temp_1',
                             'pitch_Atech_motor_temp_1'],

            'motor_temp_2': ['pitch_Atech_hub_temp_2',
                             'pitch_Atech_cabinet_temp_2',
                             'pitch_Atech_motor_current_2',
                             'pitch_position_2', 'converter_power', 'wind_speed', 'rotor_speed',
                             'pitch_Atech_capacitor_temp_2',
                             'pitch_Atech_converter_temp_2',
                             'pitch_Atech_motor_temp_2'],

            'motor_temp_3': ['pitch_Atech_hub_temp_3',
                             'pitch_Atech_cabinet_temp_3',
                             'pitch_Atech_motor_current_3',
                             'pitch_position_3', 'converter_power', 'wind_speed', 'rotor_speed',
                             'pitch_Atech_capacitor_temp_3',
                             'pitch_Atech_converter_temp_3',
                             'pitch_Atech_motor_temp_3'], }
    }

