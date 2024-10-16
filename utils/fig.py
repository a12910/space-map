import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import spacemap
import cv2
import numpy as np
    
def fig_show_box(data, columns, ylim=[0, 1.0]):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] ='sans-serif'
    fig, ax = plt.subplots()
    ax.set_ylim(*ylim)
    sns.boxplot(data)
    ax.set_xticklabels(columns)
    plt.show()
    
def fig_show_df1(df):
    npI = df[["x", "y"]].values
    spacemap.imshow(spacemap.show_img(npI))
    
def plot_hist(img):
    img = np.array(img, dtype=np.uint8)
    grayHist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue]) # x和y的范围
    plt.xlabel("gray Level")
    plt.ylabel("Number Of Pixels")
    plt.show()