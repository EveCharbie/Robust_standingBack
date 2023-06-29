# --- Open file pickle --- #
def openpickle(name_file, name_variable):
    with open(name_file, "rb") as file:
        name_variable = pickle.load(file)
