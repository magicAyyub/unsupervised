# Rapport d'Expérimentation : Apprentissage Non Supervisé

**Membres du groupe :**
* PASINI Giorgio
* NIYUNGEKO Brandon Tchaka
* DOUMBIA Ayouba

---

## Introduction

Ce projet d'apprentissage non supervisé a pour but d'évaluer, de comparer et de comprendre le comportement de différents algorithmes de compression, de quantification et de génération sur deux types de données d'images :
1. **MNIST** : Un jeu de données classique composé d'images de chiffres manuscrits en niveaux de gris de dimension $28 \times 28$ ($784$ caractéristiques).
2. **Shapes** : Un jeu de données plus complexe comprenant des formes géométriques colorées (cercles, carrés, triangles, croix, étoiles, barres) de dimension $3 \times 32 \times 32$ ($3072$ caractéristiques), avec des variations de forme, de couleur et de position.

L'analyse de ces algorithmes est structurée autour de trois grands axes :
* **La compression et la reconstruction** : Évaluer la capacité du modèle à projeter les données dans un espace latent réduit et à les reconstruire avec une distorsion minimale (mesurée par la Mean Squared Error - MSE).
* **Le codebook et l'efficacité de stockage** : Mesurer le coût de stockage partagé (paramètres fixes du modèle) et le coût de stockage individuel (représentation compressée par image), afin de calculer le ratio de compression théorique.
* **La structure de l'espace latent et la génération** : Analyser si l'espace latent appris est continu, régulier et interpolable, permettant ainsi la génération de nouveaux échantillons réalistes.

---

## 1. Algorithme de Quantification : K-Means

L'algorithme des K-Means effectue une quantification vectorielle en partitionnant les données en $K$ clusters. L'espace latent y est discret (indice du centroïde le plus proche). Le codebook contient les coordonnées des $K$ centroïdes dans l'espace des données.

### 1.1 Expérimentation sur MNIST

Sur MNIST, nous avons étudié l'impact du nombre de clusters $K$ sur l'inertie et la MSE de reconstruction.

![Exemples MNIST](report_assets/01_kmeans_cell4_out0.png)

* **Sensibilité à l'initialisation** : Nous constatons que l'écart d'inertie entre la pire et la meilleure initialisation aléatoire est de $0.70\%$, le gain obtenu après 10 lancements successifs étant de $0.53\%$. L'utilisation de l'initialisation `k-means++` stabilise grandement la convergence par rapport à une initialisation uniforme.
* **Analyse de la reconstruction** : Le centroïde représente la "moyenne" des images du cluster. Pour $K=10$, les centroïdes sont des chiffres moyennés flous. En augmentant $K$, les centroïdes se spécialisent et la reconstruction devient plus nette.

![Courbes K-Means MNIST](report_assets/01_kmeans_cell25_out0.png)

* **Performance de classification induite** : En attribuant à chaque cluster le chiffre majoritaire présent en son sein, nous évaluons la pureté des clusters :
  * Pour $K = 10$ : Précision de $56.78\%$, MSE de $0.0495$.
  * Pour $K = 64$ : Précision de $82.60\%$, MSE de $0.0370$.
  * Pour $K = 512$ : Précision de $91.25\%$, MSE de $0.0263$.

![Reconstruction K-Means MNIST](report_assets/01_kmeans_cell30_out0.png)
![Centroïdes K-Means MNIST](report_assets/01_kmeans_cell14_out0.png)
![Contingence K-Means MNIST](report_assets/01_kmeans_cell34_out0.png)

### 1.2 Expérimentation sur Shapes

Sur le jeu de données Shapes, K-Means montre des limites fondamentales liées à la nature des images colorées.

![Exemples Shapes](report_assets/01_kmeans_shapes_cell4_out0.png)
![Centroïdes Shapes K=6](report_assets/01_kmeans_shapes_cell14_out0.png)

* **Limites de la métrique Euclidienne sur pixels bruts** : Pour $K = 6$ (nombre réel de formes), les centroïdes appris ne représentent pas des formes géométriques nettes, mais des aplats de couleurs moyennes à des positions spécifiques.
* **Échec de l'association aux classes géométriques** :
  * Pour $K = 6$, la précision de classification par vote majoritaire n'est que de $26.29\%$ (très proche du hasard à $16.67\%$).
  * Même pour $K = 512$, la précision ne dépasse pas $43.50\%$.
* **Explication** : L'inertie euclidienne est dominée par les variations de couleur du fond et de la forme, ainsi que par la position (translation) de la forme. K-Means regroupe donc les images par couleur dominante ou par zone d'activation spatiale des pixels, et non par structure géométrique (cercle vs triangle).

![Reconstruction K-Means Shapes](report_assets/01_kmeans_shapes_cell30_out0.png)
![Contingence K-Means Shapes](report_assets/01_kmeans_shapes_cell34_out0.png)

---

## 2. Réduction Linéaire de Dimension : PCA (Analyse en Composantes Principales)

La PCA projette linéairement les données dans un sous-espace continu défini par les vecteurs propres de la matrice de covariance. Le codebook contient la moyenne des données et les $N$ premiers vecteurs propres.

### 2.1 Analyse de la Variance Expliquée Cumulée

Avant de choisir la dimension latente $N$, nous analysons le spectre des valeurs propres pour déterminer la part de variance conservée.

![Variance Expliquée PCA MNIST](report_assets/pca_mnist_variance.png)
![Variance Expliquée PCA Shapes](report_assets/pca_shapes_variance.png)

* **MNIST** : 
  * $90\%$ de la variance est capturée avec $N = 85$ composantes.
  * $95\%$ de la variance avec $N = 150$ composantes.
  * $99\%$ de la variance avec $N = 326$ composantes.
* **Shapes** :
  * $90\%$ de la variance est capturée avec $N = 427$ composantes.
  * $95\%$ de la variance avec $N = 711$ composantes.
  * $99\%$ de la variance avec $N = 1104$ composantes.

Le dataset Shapes requiert beaucoup plus de composantes linéaires pour atteindre un niveau de variance équivalent à MNIST, ce qui s'explique par les degrés de liberté supplémentaires (3 canaux couleur, translations, rotations).

### 2.2 Qualité de Reconstruction et Espace Latent

En fixant $N$ pour conserver environ $90\%$ de la variance ($N=87$ pour MNIST et $N=513$ pour Shapes), nous obtenons les reconstructions suivantes :

![Reconstruction PCA MNIST](report_assets/pca_mnist_reconstruction.png)
![Reconstruction PCA Shapes](report_assets/pca_shapes_reconstruction.png)

* **MNIST (N=87)** : Reconstruction très nette avec une MSE de $0.0065$. Le ratio de compression est de $8.35$.
* **Shapes (N=513)** : Reconstruction propre des formes et couleurs avec une MSE de $0.0029$. Le ratio de compression est de $4.58$.

L'espace latent 2D ($N=2$) montre une projection continue où les classes se chevauchent en partie, reflétant le fait que la PCA effectue uniquement des rotations et projections orthogonales globales sans intégrer de relations non linéaires complexes.

![Espace Latent PCA 2D MNIST](report_assets/pca_mnist_latent.png)
![Espace Latent PCA 2D Shapes](report_assets/pca_shapes_latent.png)

---

## 3. Compression Non Linéaire : Autoencoder (AE)

L'Autoencoder projette les données de manière non linéaire dans un espace latent continu via un réseau de neurones encodeur, et reconstruit les images via un décodeur. Le codebook est constitué des poids et biais du décodeur.

### 3.1 Impact de la Dimension Latente (MNIST)

Nous avons entraîné un Autoencoder MLP avec 3 couches pour l'encodeur et le décodeur, et une fonction d'activation ReLU.

![Balayage dimensions latentes AE MNIST](report_assets/03_autoencoder_cell18_out0.png)

* **Compromis Qualité / Compression** :
  * Pour $latent\_dim = 2$ : MSE de $0.0439$. Les images reconstruites sont floues mais capturent l'orientation globale.
  * Pour $latent\_dim = 16$ : MSE de $0.0152$. Les chiffres deviennent lisibles.
  * Pour $latent\_dim = 64$ : MSE de $0.0068$. La netteté est comparable à l'original.
* **Comparaison PCA vs AE** : À dimension latente équivalente faible (ex: $N=2$), l'AE non linéaire bat largement la PCA en termes de distorsion visuelle, car il est capable de plier et de tordre l'espace pour maximiser l'utilisation de ses dimensions.

### 3.2 Choix de la Perte et des Fonctions d'Activation

L'expérimentation montre que le couple (Perte, Activation de sortie) est critique pour la convergence :
* L'utilisation de la perte **MSE** avec une activation de sortie linéaire ou Sigmoid donne des reconstructions stables.
* La perte **L1** favorise des reconstructions plus contrastées au prix de gradients instables près de zéro.
* L'activation **Tanh** s'est révélée sous-optimale dans nos configurations car elle sature rapidement et ralentit fortement l'apprentissage par rapport à la **ReLU** ou **LeakyReLU**.

![Activation et Perte AE](report_assets/03_autoencoder_cell20_out0.png)

### 3.3 Application sur Shapes (MLP vs Conv)

Sur le jeu de données Shapes, l'Autoencoder linéaire/MLP rencontre des difficultés à capturer les contours nets des formes géométriques. L'intégration de couches convolutives (avec prior spatial de voisinage) permet d'obtenir des contours bien plus nets et une meilleure conservation des couleurs locales, tout en réduisant le nombre de paramètres du modèle (grâce au partage des poids des filtres).

![Reconstruction AE Shapes](report_assets/03_autoencoder_cell32_out0.png)

---

## 4. Carte Auto-Organisatrice : SOM (Self-Organizing Map)

La carte de Kohonen (SOM) est un réseau de neurones non supervisé qui projette les données sur une grille bidimensionnelle discrète en préservant la topologie. Deux neurones voisins sur la grille auront des poids proches dans l'espace des données.

### 4.1 Expérimentation sur MNIST (Grille 10x10)

Nous avons entraîné une carte SOM de $10 \times 10$ neurones sur MNIST pendant $5000$ itérations.

![Grille SOM MNIST](report_assets/som_mnist_grid.png)
![Reconstruction SOM MNIST](report_assets/som_mnist_reconstruction.png)

* **Organisation Topologique** : L'examen de la grille des poids des neurones montre une transition fluide entre les chiffres. Par exemple, les neurones représentant des `0` se situent dans un coin et se transforment graduellement en `8`, puis en `3` et en `9` à mesure qu'on se déplace sur la carte.
* **Erreur de Quantification et Topologique** :
  * Quantization Error (QE) : $5.3012$.
  * Topographic Error (TE) : $0.4085$ (indique que $40.8\%$ des échantillons ont leur deuxième neurone le plus proche non adjacent au premier sur la grille).

### 4.2 Expérimentation sur Shapes : Grille 10x10 vs 30x30

La SOM a été appliquée sur les données Shapes en comparant deux topologies de grille.

#### Grille 10x10 (100 neurones)
* **Poids de la grille** : Les neurones capturent principalement des formes floues associées à des couleurs pures (aplats rouges, bleus, verts).
* **Reconstruction** : MSE de $0.0206$, avec un excellent ratio de compression ($99.19$).

![Grille SOM Shapes 10x10](report_assets/som_shapes_grid.png)
![Reconstruction SOM Shapes 10x10](report_assets/som_shapes_reconstruction.png)

#### Grille 30x30 (900 neurones)
* **Poids de la grille** : L'augmentation de la taille de la grille permet aux neurones de se spécialiser. Les formes géométriques (triangles, losanges, croix) apparaissent très nettement sur la carte avec des contours bien définis.
* **Reconstruction** : La MSE s'améliore à $0.0162$, mais le ratio de compression s'effondre à $11.09$ en raison de la taille du dictionnaire de poids à stocker ($11$ Mo contre $1.2$ Mo pour la 10x10).

![Grille SOM Shapes 30x30](report_assets/som_shapes_grid_30.png)
![Reconstruction SOM Shapes 30x30](report_assets/som_shapes_reconstruction_30.png)

---

## 5. Modèle Génératif Régularisé : VAE (Variational Autoencoder)

Le VAE régularise l'espace latent en forçant sa distribution à s'approcher d'une distribution normale standard $\mathcal{N}(0, I)$ à l'aide d'une perte de divergence de Kullback-Leibler (KL). Cela garantit l'absence de "trous" dans l'espace latent et permet de générer des données par simple échantillonnage aléatoire.

### 5.1 Évaluation sur MNIST

Nous avons comparé un VAE et un Autoencoder standard (tous deux avec un espace latent continu de dimension 2).

![Espace Latent VAE MNIST](report_assets/05_vae_cell12_out0.png)
![Génération VAE MNIST](report_assets/05_vae_cell14_out0.png)

* **Régularisation de l'espace latent** : L'espace latent du VAE se concentre de manière très compacte autour de l'origine (moyenne proche de 0, variance proche de 1).
* **Qualité de reconstruction (Coût de la régularisation)** :
  * AE MSE : $0.0490$.
  * VAE MSE : $0.0482$.
  La perte de qualité est ici minime, mais le VAE offre la capacité unique de générer des images par échantillonnage gaussien de l'espace latent, alors que l'AE génère des images aberrantes si l'on échantillonne dans des zones vides.
* **Interpolation** : La transition entre les codes latents de deux chiffres réels produit un morphing continu et réaliste (ex: passage fluide d'un 1 à un 0).

### 5.2 Évaluation sur Shapes

Sur le dataset Shapes, la contrainte de régularisation conduit à un espace latent 2D régulier ($KL = 6.011$).

![Reconstruction VAE Shapes](report_assets/05_vae_cell30_out1.png)

Bien que la reconstruction 2D soit légèrement lissée (effet de flou inhérent à la perte MSE combinée à la contrainte KL), le modèle parvient à préserver la structure globale de l'image et permet d'interpoler continûment entre différentes couleurs et formes géométriques.

---

## 6. Modèle Génératif Adversarial : GAN (Generative Adversarial Network)

Le GAN oppose un générateur ($G$), qui crée des images à partir d'un bruit latent $z \sim \mathcal{N}(0, I)$, à un discriminateur ($D$), qui apprend à distinguer les vraies images des fausses. Il n'y a pas d'encodeur direct ($x \to z$).

### 6.1 Dynamique d'Entraînement et Saturation (MNIST)

Dans notre première configuration MLP vanilla, le discriminateur a rapidement "gagné" la partie : sa loss est tombée à 0.0, tandis que celle du générateur a divergé vers 13.5. Le discriminateur saturant, ses gradients sont devenus nuls et le générateur a cessé d'apprendre, produisant des taches informes.

* **Correctif (Affaiblir le discriminateur)** : L'ajout de couches de `Dropout(0.3)` et de fonctions `LeakyReLU(0.2)` dans le discriminateur a rétabli l'équilibre.
* **Résultat** : L'accuracy de $D$ s'est stabilisée autour de $0.5$ (l'optimum théorique du jeu de minimax) et des chiffres nets et variés sont apparus.

![Courbes GAN MNIST](report_assets/06_gan_cell4_out1.png)
![Images Générées GAN MNIST](report_assets/06_gan_cell12_out1.png)

### 6.2 Application sur Shapes : Échec du MLP et Apport du WGAN-GP Convolutif

Sur le dataset Shapes (dimension 3072), le GAN MLP vanilla a échoué.
* **Le problème du MLP** : Le modèle n'a aucun prior spatial et juge les images sur des statistiques globales de couleur. Les courbes d'équilibre semblaient parfaites (accuracy de $D$ à 0.55), mais le générateur ne produisait que des **taches de couleur informes** sans aucune géométrie.

![Taches GAN MLP Shapes](report_assets/06_gan_shapes_vanilla_cell12_out1.png)
![Courbes GAN MLP Shapes](report_assets/06_gan_shapes_vanilla_cell4_out1.png)

* **Le WGAN-GP Convolutif** : Le passage à une architecture convolutive (DCGAN) avec la perte de Wasserstein et pénalité de gradient (WGAN-GP) a résolu le problème. La distance de Wasserstein décroît de façon monotone et sert de réelle métrique de qualité. Les convolutions réintroduisent la notion de bord géométrique, permettant la génération de formes fermées (cercles, triangles).

### 6.3 Évaluation en Reconstruction (Inversion Latente)

Puisqu'un GAN ne possède pas d'encodeur, la reconstruction d'une image réelle $x$ nécessite de réaliser une **inversion latente** : une descente de gradient de 300 pas sur $z$ pour minimiser $\|G(z) - x\|^2$ à générateur figé.

#### Comparaison GAN vs Autoencoder
* **MNIST** (dimension latente 100) :
  * GAN MSE : $0.0209$ | Codebook : $5.9$ Mo.
  * AE MSE : $0.0105$ | Codebook : $1.6$ Mo.
* **Shapes (MLP)** (dimension latente 100) :
  * GAN MSE : $0.0187$ | Codebook : $15.3$ Mo.
  * AE MSE : $0.0048$ | Codebook : $13.4$ Mo.

Le GAN est systématiquement dominé par l'Autoencoder sur les deux métriques de compression : sa MSE est 2 à 4 fois supérieure et son codebook est beaucoup plus lourd, ce qui le rend inefficace pour de la pure compression de données.

#### Phénomène d'asymétrie (Mode Dropping)
Sur Shapes (WGAN-GP), nous avons observé un phénomène marquant :
* Les images générées directement via $z \sim \mathcal{N}(0, I)$ sont imparfaites et parfois bruitées.
* Les images reconstruites via l'inversion latente sont en revanche extrêmement fidèles (MSE de $0.0070$), montrant que le générateur possède la capacité de produire des formes géométriques complexes, mais que le tirage direct dans le prior ne permet pas toujours de les atteindre.
* Certaines images réelles échouent complètement à l'inversion et dégénèrent en bruit : elles n'ont pas d'antécédent dans l'espace latent. C'est le phénomène de *mode dropping* : le GAN n'étant pas contraint de couvrir tout le dataset pour tricher face au discriminateur, il ignore purement et simplement certaines configurations géométriques.

---

## 7. Synthèse Comparative Globale

Le tableau ci-dessous regroupe les performances moyennes de reconstruction et d'efficacité de stockage des différents modèles sur la base d'évaluation.

### 7.1 Tableau de comparaison sur MNIST (N=10 000 échantillons)

| Modèle | Nature Latent | Dim. Latente / Grille | Taille Codebook (octets) | Taille Latente (octets) | Ratio de Compression | MSE de Reconstruction | Capacité de Génération |
|---|---|---|---|---|---|---|---|
| **K-Means** | Discret | $K=10$ | $31\ 360$ | $10\ 000$ | $758.22$ | $0.0495$ | Non (uniquement centroïdes) |
| **K-Means** | Discret | $K=512$ | $1\ 605\ 632$ | $10\ 000$ | $19.30$ | $0.0263$ | Non (uniquement centroïdes) |
| **PCA** | Continu | $N=87$ | $275\ 968$ | $3\ 480\ 000$ | $8.35$ | **$0.0065$** | Non (espace mal régularisé) |
| **Autoencoder**| Continu | $d=2$ | $345\ 276$ | $24\ 000$ (eval) | $25.48$ | $0.0439$ | Non (trous dans l'espace) |
| **SOM** | Discret | $10 \times 10$ | $313\ 600$ | $10\ 000$ | $96.91$ | $0.0376$ | Non (grille discrète) |
| **VAE** | Continu | $d=2$ | $345\ 276$ | $24\ 000$ (eval) | $25.48$ | $0.0482$ | **Oui** (Échantillonnage gaussien) |
| **GAN (MLP)** | Continu | $d=100$ | $5\ 945\ 408$ | $400\ 000$ (eval)| $1.54$ (AE reference) | $0.0209$ | **Oui** (Nette mais incomplète) |

### 7.2 Tableau de comparaison sur Shapes (N=10 000 échantillons)

| Modèle | Nature Latent | Dim. Latente / Grille | Taille Codebook (octets) | Taille Latente (octets) | Ratio de Compression | MSE de Reconstruction | Limites et Remarques |
|---|---|---|---|---|---|---|---|
| **K-Means** | Discret | $K=6$ | $73\ 728$ | $10\ 000$ | **$1467.61$** | $0.0277$ | Groupement par couleur uniquement. |
| **K-Means** | Discret | $K=512$ | $6\ 291\ 456$ | $10\ 000$ | $19.50$ | $0.0163$ | Toujours insensible aux classes. |
| **PCA** | Continu | $N=513$ | $6\ 316\ 032$ | $20\ 520\ 000$ | $4.58$ | **$0.0029$** | Excellente qualité mais très lourd. |
| **Autoencoder**| Continu | $d=2$ | $3\ 306\ 708$ | $24\ 000$ (eval) | $11.07$ | $0.0266$ | Flou géométrique modéré. |
| **SOM** | Discret | $10 \times 10$ | $1\ 228\ 800$ | $10\ 000$ | $99.19$ | $0.0206$ | Représente des taches de couleur. |
| **SOM** | Discret | $30 \times 30$ | $11\ 059\ 200$ | $20\ 000$ | $11.09$ | $0.0162$ | Formes géométriques nettes apprises. |
| **VAE** | Continu | $d=2$ | $3\ 306\ 708$ | $24\ 000$ (eval) | $11.07$ | $0.0284$ | Génération continue par morphing. |
| **GAN (MLP)** | Continu | $d=100$ | $15\ 326\ 208$ | $4\ 000\ 000$ | $6.36$ | $0.0187$ | Échec total sur les formes. |

---

## Conclusion

Ce projet met en évidence qu'il n'existe pas d'algorithme universel d'apprentissage non supervisé ; le choix dépend du compromis recherché entre fidélité de reconstruction, taux de compression et capacité générative.

1. **Pour la compression pure** : Les méthodes discrètes comme **K-Means** et la **SOM 10x10** offrent les ratios de compression les plus élevés (supérieurs à 100 ou 1000) mais au prix d'une perte d'information importante (la reconstruction est une image moyenne).
2. **Pour la fidélité de reconstruction** : La **PCA** est extrêmement performante et rapide à calculer lorsque le nombre de composantes est grand ($N > 400$). Cependant, sa nature linéaire la rend très lourde en stockage individuel. Les **Autoencoders** (surtout convolutifs) s'imposent comme le meilleur compromis non linéaire, offrant une excellente MSE à faible dimension latente.
3. **Pour la génération de données** : Le **VAE** se distingue par la régularité mathématique de son espace latent, ce qui en fait l'outil idéal pour l'interpolation et le morphing. Le **GAN**, quant à lui, excelle dans la production d'images nettes à haute fréquence spatiale (grâce aux convolutions et au WGAN-GP), mais souffre d'instabilités à l'entraînement et de distorsions importantes lors de la reconstruction par inversion de code.
