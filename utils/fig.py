import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import spacemap
    
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
    spacemap.imshow(spacemap.show_img3(npI))
    