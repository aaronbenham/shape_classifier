import tkinter as tk
import numpy as np
import os
import tensorflow as tf
from time import sleep
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("prediction.h5")

DIR = "data/circle/"

# Set number of rows and columns
ROWS = 32
COLS = 32

# Create a grid of None to store the references to the tiles
tiles = np.zeros(ROWS*COLS).reshape(ROWS, COLS)

def draw(event):
    # Get rectangle diameters
    col_width = c.winfo_width()/COLS
    row_height = c.winfo_height()/ROWS
    
    # Calculate column and row number
    col = int(event.x//col_width)
    row = int(event.y//row_height)

    # If the tile is not filled, create a rectangle
    if not tiles[row][col]:
        tiles[row][col] = 1
        c.create_rectangle(col*col_width, row*row_height, (col+1)*col_width, (row+1)*row_height, fill="black", outline="black")
    

def delete(event):
    col_width = c.winfo_width()/COLS
    row_height = c.winfo_height()/ROWS

    col = int(event.x//col_width)
    row = int(event.y//row_height)

    if tiles[row][col]:
        tiles[row][col] = 0
        c.create_rectangle(col*col_width, row*row_height, (col+1)*col_width, (row+1)*row_height, fill="white", outline="white")

def output(event):
    count = 0
    while os.path.exists(DIR + str(count) + ".npy"):
        count += 1 
    
    np.save(DIR + str(count), tiles)


def model_prediction(event):
    shapes = np.array(["circle", "square", "triangle"])

    np.save("image_to_predict", tiles)
    img = np.load("image_to_predict.npy")

    img = img.reshape(1, 32, 32)

    res = model.predict(img)
    print(shapes[np.argmax(res[0])])


def clear_canvas(event):
    col_width = c.winfo_width()/COLS
    row_height = c.winfo_height()/ROWS

    for i in range(32):
        for j in range(32):
            tiles[i][j] = 0
            c.create_rectangle(j*col_width, i*row_height, (j+1)*col_width, (i+1)*row_height, fill="white", outline="white")


# Create the window, a canvas and the mouse click event binding
root = tk.Tk()

c = tk.Canvas(root, width=500, height=500, borderwidth=5, background='white')
c.pack()

c.bind("<B1-Motion>", draw)
c.bind("<Button-1>", draw)

c.bind("<B3-Motion>", delete)
c.bind("<Button-3>", delete)

root.bind("z", output)
root.bind("p", model_prediction)
root.bind("c", clear_canvas)

root.mainloop()