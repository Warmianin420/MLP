FOLDER_DANYCH = "analiza zdarzen"

PLIK_ZBIORCZY = r"D:/DANE_NILM/Stany_ustalone_wyselekcjonowane.xlsx"
MAINS_RAW_FILE = r"D:/DANE_NILM/dane_pqs.csv"

MODEL_NAME = "nilm_model.keras"

TEST_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 120
PATIENCE = 12

TARGET_COLS = [
    "kettle", "induction_cooker", "phone_charger", "microwave", "mixer",
    "toaster", "tv", "spin_dryer", "coffee_maker", "immersion_heater",
    "sandwich_maker", "decoder", "lamp", "aquarium", "heater",
    "usb_c_charger", "laptop", "christmas_tree", "timer",
    "hair_straightener", "fridge", "printer", "bathroom_heater", "Monitor"
]
