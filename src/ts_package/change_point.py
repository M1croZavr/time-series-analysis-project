import numpy as np


class ChangePointDetector(object):
    """
    Класс для детектирования разладок методом CUSUM

    currunt_state_observations_size - Размер временного окна текущего состояния ряда 
    
    probable_change_point_area_size - Размер окна, внутри которого ищем разладку

    threshold_coef - Порог, выше которого детектируем аномалию

    drift_coef - допустимое отклонение от среднего
    """

    def __init__(self, currunt_state_observations_size=21, probable_change_point_area_size=7, 
                 threshold_coef=4.0, drift_coef=0.5):
        self.currunt_state_observations_size = currunt_state_observations_size
        self.probable_change_point_area_size = probable_change_point_area_size
        self.threshold_coef = threshold_coef
        self.drift_coef = drift_coef

    def change_point_definition(self,ts_window_anomalies):
        '''
        если в окне свыше 70% аномалий, то это разладка
        '''

        probable_change_point_area = ts_window_anomalies[-self.probable_change_point_area_size:]
        
        if probable_change_point_area.sum() >= 0.7 * self.probable_change_point_area_size:
            return np.where(probable_change_point_area==1)[0][0]
        else:
            return -1
    
    def detect_periods(self, ts):
        '''
        Функция возвращает np.array под название periods длиной равной длине временного ряда
        periods имеет значения 1 и 0, которые сменяют друг друга в том случае, если произошла разладка
        
        ts - временной ряд
        '''
        
        time_window_size = self.currunt_state_observations_size + self.probable_change_point_area_size
        start_window_index = 0
        end_window_index = 0
        anomalies = np.zeros(len(ts))
        gp = np.zeros(len(ts))
        gn = np.zeros(len(ts))
        periods = np.zeros(len(ts))

        while end_window_index <= len(ts):
            ts_window = ts[start_window_index:end_window_index]
            
            current_state_std = np.std(ts_window)
            current_state_mean = np.mean(ts_window)
            
            drift = self.drift_coef * current_state_std
            threshold = self.threshold_coef * current_state_std

            gp[start_window_index:end_window_index] = 0
            gn[start_window_index:end_window_index] = 0
            
            for idx in range(start_window_index + 1, end_window_index):
                
                gp[idx] = max(gp[idx-1] + ts[idx] - current_state_mean - drift, 0)
                gn[idx] = min(gn[idx-1] + ts[idx] - current_state_mean + drift, 0)

                if (gp[idx] > threshold) or (gn[idx] < -threshold):
                    anomalies[idx] = 1
                
            ts_ready_anomalies = anomalies[: end_window_index]
    
            relative_change_point_start = self.change_point_definition(ts_ready_anomalies)
            
            if relative_change_point_start >= 0:
                change_point_start = start_window_index + relative_change_point_start + time_window_size - self.probable_change_point_area_size
                if periods[change_point_start] == 1:
                    periods[change_point_start:] = 0
                else:
                    periods[change_point_start:] = 1
                #двигаемся сразу в новую "статистику", так как случилась разладка
                start_window_index += self.currunt_state_observations_size
            else:
                #если разладки нет, то идем с шагом 1
                start_window_index += 1
    
            end_window_index = start_window_index + time_window_size
        
        return periods
