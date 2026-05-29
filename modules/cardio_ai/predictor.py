from modules.kanea_modules.scaffold import scaffold_predict, MODULES

MODULE_KEY = "cardio"
_CONFIG    = MODULES[MODULE_KEY]


def predict_cardio(image_path=None, params=None):
    return scaffold_predict(MODULE_KEY, image_path=image_path, params=params)
