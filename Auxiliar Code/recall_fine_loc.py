import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

data = pd.read_excel('exp2GraficasLocFina.xlsx')

losses = data['Loss']
# cloudy1 = data['Cloudy1']
# cloudy2 = data['Cloudy2']
# cloudy4 = data['Cloudy4']
# cloudy8 = data['Cloudy8']
#
# night1 = data['Night1']
# night2 = data['Night2']
# night4 = data['Night4']
# night8 = data['Night8']
#
# sunny1 = data['Sunny1']
# sunny2 = data['Sunny2']
# sunny4 = data['Sunny4']
# sunny8 = data['Sunny8']

avg1 = data['Average1']
avg2 = data['Average2']
avg4 = data['Average4']
avg8 = data['Average8']

k_list = [1, 2, 4, 8]
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']


def create_plot(ilum, ilum1, ilum2, ilum4, ilum8):
    plt.figure(figsize=(9, 5), dpi=120)
    for i in range(0, len(losses)):
        loss = losses[i]
        plt.scatter(1, ilum1[i], color=colors[i], label=loss)
        plt.scatter(2, ilum2[i], color=colors[i])
        plt.scatter(4, ilum4[i], color=colors[i])
        plt.scatter(8, ilum8[i], color=colors[i])
        plt.plot([1, 2], [ilum1[i], ilum2[i]], colors[i])
        plt.plot([2, 4], [ilum2[i], ilum4[i]], colors[i])
        plt.plot([4, 8], [ilum4[i], ilum8[i]], colors[i])

    plt.axis([0.9, 8.1, 0, 100])
    plt.ylabel('Average Recall @K (%)', fontsize=16)
    plt.xlabel('K: Nearest neighbours number', fontsize=16)
    # plt.suptitle('Fine Localization', fontsize=24)
    # plt.title('Illumination condition: ' + ilum, fontsize=20)
    plt.title(f'Experiment 1, Fine localization', fontsize=18)
    # plt.title(f'Experiment 1, Fine localization, Illumination: {ilum}', fontsize=18)
    plt.legend(fontsize=13)
    plt.grid()
    plt.savefig("exp1RecallFineLoc" + ilum + ".png")
    return


create_plot("Average", avg1, avg2, avg4, avg8)

# create_plot("Cloudy", cloudy1, cloudy2, cloudy4, cloudy8)
# create_plot("Night", night1, night2, night4, night8)
# create_plot("Sunny", sunny1, sunny2, sunny4, sunny8)

