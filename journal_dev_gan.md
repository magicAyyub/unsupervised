# 2026-07-16 14:00, cadrage du GAN

Debut de la partie GAN. Premiere question tranchee avant d'ecrire du code : est-ce que le GAN
herite de `BaseModel` comme les autres ?

Non. Un GAN n'a pas d'encodeur, il n'existe aucune fonction `x -> z`. Le generateur fait `z -> x`,
ce qui ressemble a un `decode`, mais `encode(X)` n'a mathematiquement pas de sens. Heriter du
contrat nous obligerait a lever `NotImplementedError` sur une methode de l'interface, donc a
casser le contrat a l'execution. La classe reste autonome : `fit` / `generate` / `get_codebook`.

Choix de depart, volontairement simple : MLP (pas de convolutions), MNIST d'abord, shapes ensuite.

# 2026-07-16 15:30, premiere version, elle ne marche pas

Premiere config : latent 64, G `[256, 512]`, D `[512, 256]`, 40 epochs sur 15000 images.
Resultat : des taches informes, aucun chiffre reconnaissable.

Les courbes disent exactement ce qui se passe :

- loss D : 1.15 -> 0.44 -> 0.20 -> 0.10 -> 0.0
- loss G : 1.09 -> 3.94 -> 3.94 -> 9.61 -> 13.51

`D` gagne la partie de facon ecrasante. Il separe le vrai du faux trop facilement, ses logits
saturent, et le gradient qui doit remonter jusqu'a `G` devient nul. `G` n'apprend plus rien.

Erreur suivante, par reflexe : j'ai grossi le generateur (`[256, 512, 1024]`, latent 100, 60 epochs,
60000 images) en pensant qu'il etait trop faible. **Ca a empire.** loss D a 0.0 et loss G a 13.5.
Logique retrospectivement : le probleme n'etait pas la capacite de `G`, c'etait le desequilibre.
Un `G` plus gros face a un `D` qui gagne deja ne change rien, `G` ne recoit toujours aucun gradient.

# 2026-07-16 16:15, le correctif, contre-intuitif

Il fallait **affaiblir `D`**, pas renforcer `G`. Ajout de `Dropout(0.3)` entre les couches cachees
du discriminateur, plus LeakyReLU(0.2) partout (y compris dans `G`).

Resultat immediat, meme budget d'entrainement :

- loss D : 1.16 -> 1.01 -> 1.20 -> 1.23 -> 1.26 -> 1.28
- loss G : 0.98 -> 1.38 -> 1.04 -> 0.99 -> 0.94 -> 0.92

Les deux courbes se croisent vers l'epoch 12 puis plateau. Partie equilibree, et des chiffres
reconnaissables et varies apparaissent. L'ecart-type inter-images passe de 0.116 a 0.175, contre
0.193 pour les vraies donnees : pas de mode collapse.

La lecon a retenir, et elle est generale : sur un GAN, quand la generation est mauvaise, le premier
reflexe ne doit pas etre de renforcer `G`. C'est souvent `D` qu'il faut handicaper. Corollaire, une
loss qui descend n'est pas un bon signe ici : les deux reseaux ont des objectifs opposes, la somme
ne peut pas tendre vers zero. Une loss de `D` qui s'effondre est une alerte, pas un succes.

# 2026-07-16 17:00, Compression, Projection
.

Ajout de deux detours, avec leur mesure :

- **Compression** : `invert()`, une descente de gradient sur `z` a `G` fige, pour minimiser
  `||G(z) - x||`. Ce n'est pas un encodeur et le nommage doit rester honnete : l'AE code en une
  passe avant, ici il faut 300 pas de gradient.
- **Projection** : `extract_features()`, la derniere couche cachee de `D` recyclee en extracteur.
  C'est l'usage historique de Radford et al. (DCGAN, 2016).

# 2026-07-16 17:45, resultats des 3 axes

Les mesures tranchent, et deux d'entre elles sont negatives. Elles restent dans le notebook :
un resultat negatif mesure vaut mieux qu'un axe elude.

**Compression** (1000 images, latent 100, face a un AE de dimension latente egale) :

| | GAN | AutoEncoder |
|---|---|---|
| MSE | 0.0209 | 0.0105 |
| codebook | 5 945 408 o | 1 632 336 o |
| ratio | 0.49 | 1.54 |

Le GAN est 2x pire en reconstruction et son ratio est **inferieur a 1** : il gonfle les donnees.
Les poids de `G` pesent plus que les 1000 images d'origine. Ce cout est fixe et s'amortit : le
seuil de rentabilite est a ~2170 images, le plafond theorique a 7.84 (784/100).

Fait plus interessant que les chiffres : a l'inversion, **certains chiffres reels echouent** et
degenerent en tache. Ces images n'ont aucun antecedent dans l'espace latent, `G` ne sait pas les
produire quel que soit `z`. Un AE reconstruit toujours quelque chose, meme flou, parce qu'il a ete
entraine a couvrir toutes les donnees. Le GAN n'a jamais eu cette contrainte : il lui suffit de
tromper `D`, donc il peut ignorer une partie du dataset sans etre penalise.

**Projection** (kNN sur 2 composantes PCA, 3000 images) :

- pixels bruts : 0.441
- features de `D` : 0.430

Les features de `D` font **moins bien que les pixels bruts**, et le nuage est visiblement plus
informe. Logique une fois enonce : `D` apprend a separer "reelle" de "fausse", pas a separer les
chiffres. Rien dans sa loss ne l'encourage a distinguer un 3 d'un 8, les deux etant de vraies
images. Le detour est un echec, et c'est un echec mesure.

Conclusion d'etape : le GAN est le premier modele du projet qui sacrifie deliberement 2 axes sur 3.
La contrepartie est reelle, sur l'axe generation il produit des images nettes la ou l'AE rend du
flou (l'AE minimise une MSE, donc il moyenne en cas de doute, et une moyenne d'images est floue ;
le GAN doit seulement etre credible aux yeux de `D`, et une image moyennee ne trompe personne).

