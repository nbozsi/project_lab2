import os
from glob import glob

path = "data/kiegyenlito_arak"

for folder in os.listdir(path):
    ctr = 0
    excels = os.listdir(os.path.join(path, folder))
    neg_excels = sorted(filter(lambda x: x.startswith("Neg_KE"), excels))
    poz_excels = sorted(filter(lambda x: x.startswith("Poz_KE"), excels))

    for x in poz_excels[:-1]:
        os.remove(os.path.join(path, folder, x))

    for x in neg_excels[:-1]:
        os.remove(os.path.join(path, folder, x))
