import matplotlib
matplotlib.use("TkAgg")

import gymnasium as gym
import gym_anytrading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from gymnasium.envs.registration import register

#  Enregistrement de l'environnement
try:
    register(
        id="stocks-v0",
        entry_point="gym_anytrading.envs:StocksEnv",
        kwargs={"df": pd.DataFrame(), "window_size": 10, "frame_bound": (10, 10)}
    )
    print(" Environnement 'stocks-v0' enregistré avec succès !")
except Exception as e:
    print("⚠️ Environnement déjà enregistré :", e)

#  Génération des données boursières
np.random.seed(42)
data_size = 500

# GOLD (Actif stable)
gold_prices = np.cumsum(np.random.randn(data_size) * 2 + 1000)
df_gold = pd.DataFrame(gold_prices, columns=["Close"])

# ⚡ BITCOIN (Actif très volatil)
bitcoin_prices = np.cumsum(np.random.randn(data_size) * 200 + 30000)
df_bitcoin = pd.DataFrame(bitcoin_prices, columns=["Close"])

# Chargement des environnements (INVERSION DES ALGORITHMES)
env_dqn = gym.make("stocks-v0", df=df_bitcoin, window_size=10, frame_bound=(10, len(df_bitcoin)))
env_ppo = gym.make("stocks-v0", df=df_gold, window_size=10, frame_bound=(10, len(df_gold)))

#Chargement des modèles
dqn_model = DQN.load("dqn_model.zip")
ppo_model = PPO.load("ppo_model.zip")

def plot_trading_strategy(models, envs, titles, colors, assets):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Comparaison des stratégies de Trading : DQN vs PPO", fontsize=16, fontweight='bold')

    for i, (model, env, title, color, asset) in enumerate(zip(models, envs, titles, colors, assets)):
        try:
            obs, _ = env.reset()
        except ValueError:
            obs = env.reset()

        prices, steps, actions = [], [], []

        for step in range(len(asset) - 10):
            action, _ = model.predict(obs, deterministic=True)
            try:
                obs, _, done, _, _ = env.step(action)
            except ValueError:
                obs, _, done, _ = env.step(action)

            if step % 5 == 0:
                prices.append(asset["Close"].iloc[step + 10])
                actions.append(action)
                steps.append(step)

            if done:
                try:
                    obs, _ = env.reset()
                except ValueError:
                    obs = env.reset()

        ax = axes[i]
        scatter = ax.scatter(steps, prices, c=actions, cmap=color, s=5, alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Temps")
        ax.set_ylabel("Prix du marché")
        ax.grid(True, linestyle="--", alpha=0.5)

        if "BITCOIN" in title:
            ax.set_yscale("log")

        # 🔥 Annotation des actifs
        asset_name = "BITCOIN" if "DQN" in title else "GOLD"
        ax.annotate(f"Actif: {asset_name}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, fontweight='bold', color="black")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Type d'Action (Achat/Vente)")

    plt.tight_layout()
    plt.show()
    plt.pause(1)

plot_trading_strategy(
    [dqn_model, ppo_model],
    [env_dqn, env_ppo],
    ["Stratégie DQN (BITCOIN)", "Stratégie PPO (GOLD)"],  # ✅ Inversion ici
    ["coolwarm", "plasma"],
    [df_bitcoin, df_gold]
)

