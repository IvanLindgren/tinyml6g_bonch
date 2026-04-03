import numpy as np

# Глобальные настройки (импортируются из визуализатора)
SCALE_GYR = 0.5  
DT = 0.01
DECAY = 0.95  

class BaseParser:
    def parse_payload(self, vals, client_dict):
        raise NotImplementedError

class LsmParser(BaseParser):
    def parse_payload(self, vals, c):
        if len(vals) != 12: return False
        c['latest_real'] = vals[0:6]
        c['latest_pred'] = vals[6:12]
        
        # Интеграция 6G Бимформинга (Гироскопы)
        real_dx = c['latest_real'][4] * SCALE_GYR * DT  
        real_dy = c['latest_real'][5] * SCALE_GYR * DT  
        c['real_pos'][0] = (c['real_pos'][0] + real_dx) * DECAY
        c['real_pos'][1] = (c['real_pos'][1] + real_dy) * DECAY
        
        pred_dx = c['latest_pred'][4] * SCALE_GYR * DT
        pred_dy = c['latest_pred'][5] * SCALE_GYR * DT
        c['pred_pos'][0] = (c['pred_pos'][0] + pred_dx) * DECAY
        c['pred_pos'][1] = (c['pred_pos'][1] + pred_dy) * DECAY
        
        c['abs_real_pos'][0] += c['latest_real'][4] * DT
        c['abs_real_pos'][1] += c['latest_real'][5] * DT
        c['abs_pred_pos'][0] += c['latest_pred'][4] * DT
        c['abs_pred_pos'][1] += c['latest_pred'][5] * DT
        return True

class AdxlParser(BaseParser):
    def parse_payload(self, vals, c):
        if len(vals) != 12: return False
        c['latest_real'] = vals[0:6]
        c['latest_pred'] = vals[6:12]
        
        # Для ADXL сымитируем координату "мыши" через наклоны акселерометра
        c['real_pos'][0] = c['latest_real'][0] * 5.0 
        c['real_pos'][1] = c['latest_real'][1] * 5.0 
        c['pred_pos'][0] = c['latest_pred'][0] * 5.0
        c['pred_pos'][1] = c['latest_pred'][1] * 5.0
        
        c['abs_real_pos'][0] = c['real_pos'][0]
        c['abs_real_pos'][1] = c['real_pos'][1]
        c['abs_pred_pos'][0] = c['pred_pos'][0]
        c['abs_pred_pos'][1] = c['pred_pos'][1]
        return True

sensor_parsers = {
    "LSM": LsmParser(),
    "ADXL": AdxlParser()
}
