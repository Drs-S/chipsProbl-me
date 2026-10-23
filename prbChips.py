import numpy as np
import matplotlib.pyplot as plt

# Paramètres du scénario
list_gout_chips = ["barbecue", "nature", "sel&vinaigre", "paprika", "crème&oignons"]
nombre_soiree = 100
nombre_ami = 3
nb_simulations = 200

# 1. Phase interactive : Collecte des choix réels des utilisateurs
print("--- PHASE DE TEST (5 soirées d'évaluation) ---")
votes_cumules = np.zeros(len(list_gout_chips))

for i, gout in enumerate(list_gout_chips):
    print(f"\nSoirée {i+1} - Dégustation des chips : {gout.upper()}")
    votes_positifs = 0
    for ami in range(1, nombre_ami + 1):
        while True:
            reponse = input(f"Ami {ami}, as-tu aimé ce goût ? (o/n) : ").strip().lower()
            if reponse in ['o', 'oui', 'y', 'yes']:
                votes_positifs += 1
                break
            elif reponse in ['n', 'non', 'no']:
                break
            else:
                print("Réponse invalide, tapez 'o' pour oui ou 'n' pour non.")
    votes_cumules[i] = votes_positifs

# Calcul des vraies probabilités basées sur vos votes
probabilites_reelles = votes_cumules / nombre_ami
meilleur_gout_idx = np.argmax(probabilites_reelles)

print("\n--- RÉSULTATS DES VOTES ---")
for gout, proba in zip(list_gout_chips, probabilites_reelles):
    print(f"{gout.capitalize()} : {proba*100:.0f}% d'approbation")
print(f"Le goût favori du groupe est donc : {list_gout_chips[meilleur_gout_idx].upper()}")


# 2. Phase de simulation des algorithmes
def argmax_aleatoire(valeurs, generateur):
    est_maximum = valeurs == valeurs.max(axis=1, keepdims=True)
    return np.argmax(est_maximum * generateur.random(valeurs.shape), axis=1)

class AlgorithmeBandit:
    def __init__(self, nb_bras):
        self.nb_bras = nb_bras
        self.generateur = np.random.default_rng()
        self.nb_tirages = np.zeros((nb_simulations, nb_bras))
        self.moyennes = np.zeros((nb_simulations, nb_bras))
        self.lignes = np.arange(nb_simulations)

    def observer(self, bras, recompenses):
        self.nb_tirages[self.lignes, bras] += 1
        ecart = recompenses - self.moyennes[self.lignes, bras]
        self.moyennes[self.lignes, bras] += ecart / self.nb_tirages[self.lignes, bras]

class Aleatoire(AlgorithmeBandit):
    def choisir(self, t):
        return self.generateur.integers(self.nb_bras, size=nb_simulations)

class EpsilonGreedy(AlgorithmeBandit):
    def __init__(self, nb_bras, epsilon=0.1):
        super().__init__(nb_bras)
        self.epsilon = epsilon

    def choisir(self, t):
        exploitation = argmax_aleatoire(self.moyennes, self.generateur)
        exploration = self.generateur.integers(self.nb_bras, size=nb_simulations)
        explore = self.generateur.random(nb_simulations) < self.epsilon
        return np.where(explore, exploration, exploitation)

class UCB(AlgorithmeBandit):
    def choisir(self, t):
        Nt_a = np.maximum(self.nb_tirages, 1e-5)
        exploration = np.sqrt(2 * np.log(t) / Nt_a)
        valeurs_ucb = np.where(self.nb_tirages == 0, np.inf, self.moyennes + exploration)
        return argmax_aleatoire(valeurs_ucb, self.generateur)

def simuler_algo(algo, nom_algo, nb_soirees):
    taux_meilleur = np.zeros(nb_soirees)
    recompense_cumulee = np.zeros(nb_soirees)
    total = np.zeros(nb_simulations)
    probas = np.tile(probabilites_reelles, (nb_simulations, 1))

    for t in range(1, nb_soirees + 1):
        bras = algo.choisir(t)
        votes_positifs = algo.generateur.binomial(nombre_ami, probas[algo.lignes, bras])
        recompenses_norm = votes_positifs / nombre_ami
        algo.observer(bras, recompenses_norm)

        total += recompenses_norm
        taux_meilleur[t - 1] = np.mean(bras == meilleur_gout_idx)
        recompense_cumulee[t - 1] = total.mean()

    pas = np.arange(1, nb_soirees + 1)
    regret = pas * probabilites_reelles.max() - recompense_cumulee
    return pas, taux_meilleur, recompense_cumulee, regret

print(f"\nSimulation des algorithmes sur les {nombre_soiree} soirées restantes...")

algos = {
    "Aléatoire": Aleatoire(len(list_gout_chips)),
    "Epsilon-Greedy (10%)": EpsilonGreedy(len(list_gout_chips), 0.1),
    "UCB": UCB(len(list_gout_chips))
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titres = ["Taux choix du meilleur goût", "Récompense moyenne", "Regret cumulé"]

for nom, algo in algos.items():
    pas, taux, rec, reg = simuler_algo(algo, nom, nombre_soiree)
    axes[0].plot(pas, taux, label=nom)
    axes[1].plot(pas, rec, label=nom)
    axes[2].plot(pas, reg, label=nom)

for i in range(3):
    axes[i].set(xlabel="Soirées", title=titres[i])
    axes[i].legend()
    axes[i].grid(alpha=0.3)
axes[0].set_ylim(0, 1.05)

plt.tight_layout()
plt.show()