from modules.kanea_modules.scaffold import scaffold_predict, MODULES

MODULE_KEY = "osteo"
_CONFIG    = MODULES[MODULE_KEY]


def predict_osteo(image_path=None, params=None):
    return scaffold_predict(MODULE_KEY, image_path=image_path, params=params)
