TOKEN = ""

# STATES
STATES = {"INITIAL": "initial",
          "LOADING": "loading",
          "PREDICT": "predict",
          "RESULTS": "result",
          "HELP": "help",
          "ERROR": "error",
          "NO_IMAGE": "no_image"}

# MESSAGES
WELCOME_MESSAGE = "This bot uses deep learning model that can classify" \
                  "different cloud types and" \
                  "predict rainfall using only cloud images." \
                  "\n\nEnter cloud image to start predicting!"

# Help Messages
HELP_MESSAGES = {"initial": "HELP: Enter '/start' to initiate the bot",
                 "loading": "HELP: Load an image to predict"}

# Error Messages
ERROR_MESSAGES = {"error": "ERROR: Unknown command, enter 'help' for more information",
                  "no_image": "ERROR: Load an image"}

# Models Paths
MODEL_PATH = "./model/best_model.pt"

# Prediction Labels
# {'As': 0, 'Ac': 1, 'Ci': 2, 'Cc': 3, 'St': 4, 'Sc': 5, 'Cb': 6, 'Cs': 7, 'Cu': 8, 'Ns': 9, 'Ct': 10}
LABELS = ["Altostratus","Altocumulus",
          "Cirrus","Cirrostratus",
          "Stratus","Stratocumulus",
          "Cumulonimbus","Cirrostratus",
          "Cumulus","Nimbostratus",
          "Contrail"]

RAIN_PRECIPITATION = ["Rain or Snow","May Produce Light Showers.",
                      "No Precipitation","No Precipitation",
                      "Drizzle","Drizzle",
                      "Showers or Snow","No Precipitation",
                      "Showers or Snow","Heavier Intensity Rain or Snow",
                      "No Precipitation"]