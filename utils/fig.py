import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
    
def fig_show_box(data, columns, ylim=[0, 11]):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] ='sans-serif'
    fig, ax = plt.subplots()
    ax.set_ylim(*ylim)
    sns.boxplot(data)
    ax.set_xticklabels(columns)
    plt.show()
    
