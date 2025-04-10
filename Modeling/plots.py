import pandas as pd
import matplotlib.pyplot as plt
plot_df = pd.read_csv("SeasonData/Seed Results.csv")

plt.figure(figsize=(8, 16))

plt.subplot(3, 2, 1)
plt.bar(plot_df["SEED"], plot_df["R32"], label="Round of 64 Wins", color="blue")
plt.xlabel("Seed")
plt.ylabel("Games Won")  
plt.title("Round of 64 Games Won By Seed")
plt.legend()


plt.subplot(3, 2, 2)
plt.bar(plot_df["SEED"], plot_df["S16"], label="Round of 32 Wins", color="blue")
plt.xlabel("Seed")
plt.ylabel("Games Won")
plt.title("Round of 32 Games Won By Seed")
plt.legend()

# plt.subplots_adjust(wspace=0.5)
plt.subplot(3, 2, 3)
plt.bar(plot_df["SEED"], plot_df["E8"], label="Sweet 16 Wins", color="blue")
plt.xlabel("Seed")
plt.ylabel("Games Won")
plt.title("Sweet 16 Games Won By Seed")
plt.legend()

plt.subplot(3, 2, 4)
plt.bar(plot_df["SEED"], plot_df["F4"], label="Elite 8 Wins", color="blue")
plt.xlabel("Seed")
plt.ylabel("Games Won")
plt.title("Elite 8 Games Won By Seed")
plt.legend()

plt.subplot(3, 2, 5)
plt.bar(plot_df["SEED"], plot_df["F2"], label="Final 4 Wins", color="blue")
plt.xlabel("Seed")
plt.ylabel("Games Won")
plt.title("Final 4 Games Won By Seed")
plt.legend()

plt.subplot(3, 2, 6)
plt.bar(plot_df["SEED"], plot_df["CHAMP"], label="Final 2 Wins", color="blue")
plt.xlabel("Seed")
plt.ylabel("Games Won")
plt.title("Final 2 Games Won By Seed")
plt.legend()

# plt.subplot(4, 2, 7)
# plt.bar(plot_df["SEED"], plot_df["CHAMP"], label="Championship Wins", color="blue")
# plt.xlabel("Seed")
# plt.ylabel("Games Won")
# plt.title("Championship Games Won By Seed")
# plt.legend()

plt.subplots_adjust(hspace=0.8)
plt.savefig("Figures/Seed Wins By Round.png")
plt.show()


plt.close()
