import sys
sys.path.append("D:\\bourg\Documents\GitHub\\artificial-intelligence-graph-search\\1-recherche-v3\src\search.py")  # Replace "/path/to/folder" with the actual path to the directory containing search.py
#print(sys.path) line by line
for p in sys.path:
    print(p)
import cv2
from lle import World
# Create an instance of the World class
#TypeError: World.__new__() missing 1 required positional argument: 'map_str'
# map is in cartes/1_agent/vide
read_file = open("cartes/1_agent/vide", "r")
map_str = read_file.read()
read_file.close()
# 

print("map_str: \n", map_str)
world_instance = World(map_str)
# img = World.get_image()
img = world_instance.get_image()
# img = World.get_image("cartes/1_agent/vide")
# img = World.get_image(D:\bourg\Documents\GitHub\artificial-intelligence-graph-search\1-recherche-v3\cartes\1_agent\impossible)
# img = World.get_image("D:\\bourg\\Documents\\GitHub\\artificial-intelligence-graph-search\\1-recherche-v3\\cartes\\1_agent\\impossible")
cv2.imshow("Visualisation", img)
# Utilisez waitKey avec 0 pour bloquer et attendre que l'utilisateur appuie sur 'enter'ou avec 1 pour continuer dans le code.
cv2.waitKey(0) # Attend que l'utilisateur appuie sur 'enter'
cv2.waitKey(1) # continue l'exécution du code