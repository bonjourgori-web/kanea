from modules.kanea_modules.scaffold import scaffold_predict, MODULES

MODULE_KEY = "gastro"
_CONFIG    = MODULES[MODULE_KEY]


def predict_gastro(image_path=None, params=None):
    return scaffold_predict(MODULE_KEY, image_path=image_path, params=params)
