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

Conclusion d'etape : le GAN sacrifie 2 taches sur 3 pour se specialiser sur la generation.
La contrepartie est reelle, sur cette tache il produit des images nettes la ou l'AE rend du
flou (l'AE minimise une MSE, donc il moyenne en cas de doute, et une moyenne d'images est floue ;
le GAN doit seulement etre credible aux yeux de `D`, et une image moyennee ne trompe personne).

# 2026-07-16 18:30, relecture des notes de cours

Repasse l'implementation au crible des points evoques en cours. Deux manques reels.

**1. Gel des poids, dans les deux sens.** Le point est explicite : quand `D` s'entraine, `G` ne
bouge pas (`.detach()`), **et inversement**. Je faisais bien le `.detach()` mais pas le gel de `D`
pendant le pas de `G`.

Mon code n'etait pas faux pour autant, et c'est ce qui rend l'erreur sournoise : l'optimiseur de
`G` ne contient que les poids de `G`, donc `D` n'etait jamais mis a jour a tort. Mais le backward
de `G` accumulait quand meme des gradients dans les poids de `D`, effaces ensuite par le
`zero_grad()` du pas suivant. Correct par ordonnancement seulement, donc fragile, et du calcul
gaspille. Ajout de `set_requires_grad(discriminator, False)` autour du pas de `G` : le gradient
traverse toujours `D` pour atteindre `G`, il n'est simplement plus accumule sur ses poids. Verifie
que `D` est bien redegele en sortie de `fit`.

**2. Metriques d'equilibre.** Le critere donne en cours : accuracy de `D` a 0.5 et grande variance
du generateur. Je ne tracais ni l'une ni l'autre, je me contentais des loss. Ajout de
`metric_history` avec les deux, relevees par epoch. L'accuracy de `D` se cale bien autour de 0.5.

Points deja couverts, verifies : GAN vanilla avec binary cross-entropy (`BCEWithLogitsLoss`),
surveillance du mode collapse, et **saturation**. Ce dernier meritait d'etre explicite dans le
notebook : la formulation d'origine fait minimiser `log(1 - D(G(z)))` a `G`, dont la derivee tend
vers 0 des que `D` rejette la fausse image avec confiance. `G` cesse d'apprendre exactement quand
il en a le plus besoin. J'utilisais deja la forme non saturante (`G` maximise `log D(G(z))`, soit
`BCE(D(fake), 1)`) mais sans le dire. C'est le mecanisme derriere l'echec du 15:30.

Sur le sampling : batchs melanges (`shuffle=True`), sinon des batchs tries par classe donneraient a
`D` une statistique de lot a exploiter plutot que de juger chaque image ; `drop_last=True` pour ne
pas fausser les moyennes de fin d'epoch avec un batch incomplet.

Reste a faire : rejouer le protocole sur le dataset shapes couleur (32x32x3 = 3072 dimensions). Le
MLP risque d'y converger moins bien, ce sera peut-etre l'argument pour passer en DCGAN convolutif.

# 2026-07-17 09:30, shapes : le MLP echoue, et les metriques ne le disent pas

Rejoue la config MNIST telle quelle sur shapes (10000 images du pipeline partage, `data_dim=3072`,
latent 100, 60 epochs). Verifie d'abord que le lot est bien celui du groupe : effectifs par forme
`[1610, 1691, 1686, 1656, 1726, 1631]`, identiques a ceux de `01_kmeans_shapes.ipynb`.

Toutes les metriques mises en place le 16/07 au soir disent que la partie est **saine**, et mieux
que sur MNIST :

- accuracy de `D` : 0.549 (contre 0.63 sur MNIST, 0.5 vise)
- loss D 1.37 / loss G 0.74, les deux plateau
- variance de `G` : 0.002 -> 0.123, ecart-type inter-images 0.160 contre 0.177 au reel

Et pourtant les images sont des **taches de couleur informes**. Aucune geometrie, aucun bord franc,
alors que les vraies images sont des polygones nets sur fond sombre.

C'est la lecon de cette entree : **les metriques d'equilibre ne mesurent pas la qualite**. Elles
disent que `D` et `G` sont a egalite, pas que `G` produit quelque chose de bon. Ici l'egalite est
obtenue par le bas. Le vrai signal d'alarme est ailleurs : `D` n'atteint que 0.547 d'accuracy sur
des faux qu'un humain rejette au premier coup d'oeil. Ce n'est pas `G` qui est trop bon, c'est `D`
qui est **aveugle**. Un MLP voit 3072 pixels sans voisinage : il n'a aucun moyen de representer un
bord. Il juge donc sur des statistiques de couleur, et `G` n'a qu'a les reproduire pour gagner.

Controles, pour ne pas conclure trop vite (le reflexe du 15:30, "c'est un desequilibre") :

| variante | accuracy D | ecart-type genere | images |
|---|---|---|---|
| dropout 0.3, 60 ep (ref) | 0.549 | 0.160 | taches |
| dropout 0.0, 60 ep | 0.575 | 0.128 | taches |
| dropout 0.1, 60 ep | 0.562 | 0.150 | taches |
| dropout 0.3, 150 ep | 0.555 | 0.172 | taches |

Ni retirer le handicap de `D`, ni multiplier l'entrainement par 2.5 ne fait apparaitre une forme.
Le probleme n'est donc pas l'equilibre et n'est pas la duree : il est **representationnel**. La
correction qui a sauve MNIST (affaiblir `D`) ne s'applique pas ici, elle aggrave meme la diversite.

# 2026-07-17 10:15, convolutions : la geometrie revient, l'equilibre casse

Passage a un `G` / `D` convolutifs (DCGAN, Radford et al. 2016), loss BCE inchangee. 60 epochs.

Effet immediat sur les images : bords francs, aplats de couleur unis, fond noir, et meme les
petits points blancs parasites du dataset reproduits. Le prior spatial etait bien ce qui manquait.

Mais l'equilibre s'effondre, exactement la pathologie du 16/07 15:30 :

- accuracy de `D` : 0.936 -> 0.998
- loss G : 5.53 -> 6.25 (diverge), loss D 0.21

Et a 200 epochs c'est **pire** (accuracy 0.998, loss G 6.25) : plus on entraine, plus `D` gagne.
Les formes restent amorphes, `G` ne recoit plus de gradient exploitable.

Applique le correctif MNIST : `Dropout2d` dans le `D` convolutif.

| variante | accuracy D | ecart-type genere | images |
|---|---|---|---|
| conv, dropout 0.0, 60 ep | 0.962 | 0.171 | bords francs, formes amorphes |
| conv, dropout 0.3, 60 ep | 0.900 | 0.148 | plus flou |
| conv, dropout 0.5, 60 ep | 0.811 | 0.152 | franchement flou, plus de bords |

**Le correctif MNIST se retourne ici.** Le dropout ramene bien l'accuracy vers 0.8, mais il detruit
les detecteurs de bords qui venaient justement de nous donner la structure. On rachete de
l'equilibre en perdant ce qu'on cherchait. Impasse : avec une loss BCE, soit `D` voit et ecrase
`G`, soit on l'aveugle et il n'y a plus de geometrie.

# 2026-07-17 11:00, WGAN-GP, et pourquoi seulement maintenant

Le cours propose WGAN, WGAN-GP, Progressive Growing, MSG-GAN. Je n'y suis pas alle d'office : il
fallait d'abord une pathologie mesuree a laquelle repondre. Elle est la, et elle est precise.

Le probleme du BCE est la **saturation** : `D` sort une probabilite, et quand il separe
parfaitement (0.998), ses logits saturent et le gradient vers `G` meurt. Le critique de Wasserstein
sort un **score non borne** : l'ecart entre vrai et faux reste exploitable quel que soit son niveau,
donc `G` recoit toujours un gradient. C'est exactement ce qui manquait. La contrainte de
1-Lipschitz passe par la **penalite de gradient** et non par le clipping des poids du slide 2 : le
clipping ampute la capacite du critique, ce qui nous ramenerait au probleme du dropout.

Detail qui compte : **pas de BatchNorm dans le critique**, LayerNorm a la place. La penalite est
definie echantillon par echantillon, une BatchNorm melangerait le lot et invaliderait la contrainte.

Resultat, 250 epochs, `n_critic = 5`, `lambda = 10`, lr 1e-4, betas (0.5, 0.9) :

- distance de Wasserstein : 35.1 -> 5.1 (ep 20) -> 4.3 (ep 130) -> 3.87, puis plateau
- variance de `G` stable a 0.137, ecart-type inter-images 0.173 contre 0.177 au reel

Gain qui n'est pas que cosmetique : **la courbe descend et veut dire quelque chose**. Avec le BCE,
une loss qui descend etait une alerte (entree du 16/07 16:15) ; ici la distance de Wasserstein est
une vraie mesure de l'ecart entre les deux distributions, elle decroit quand les images
s'ameliorent et elle plateau quand `G` a fini de progresser. On peut enfin decider de la duree
d'entrainement sur un chiffre plutot qu'a l'oeil.

Les formes deviennent lisibles (triangles, etoiles, objets pleins), sans etre propres. A 250 epochs
la distance plateau : ce n'est plus la duree qui limite.

# 2026-07-17 11:40, deux bugs trouves en branchant le convolutif sur l'API existante

Le `G` convolutif utilise des BatchNorm, et cela casse deux hypotheses que le MLP masquait.

**1. `decode()` n'etait pas une fonction de `z`.** `generate` / `decode` / `invert` laissaient `G`
en mode `train()`, donc les BatchNorm utilisaient les statistiques **du lot courant**. Mesure :
`|G(z seul) - G(z dans un lot de 64)| = 1.47` sur une echelle [-1, 1]. L'image dependait des autres
codes envoyes avec elle, ce qui vide de son sens toute la tache de compression (un recepteur qui
decode un code a la fois n'obtiendrait pas la meme image). Correctif : `eval()` dans `generate`,
`decode` et `invert`. Ecart apres correctif : 2e-6.

**2. `get_codebook()` sous-comptait.** Il iterait sur `parameters()`, ce qui **omet les buffers**
des BatchNorm (`running_mean`, `running_var`) : 899 valeurs, 3596 octets. C'est peu, mais ces
valeurs sont indispensables pour decoder, donc elles font partie du codebook. Correctif : iterer
sur `state_dict()`.

Les deux correctifs sont des **no-op sur le chemin MNIST**, verifie : le `G` MLP n'a aucun buffer,
`parameters()` et `state_dict()` couvrent le meme jeu de tenseurs, et `train()` / `eval()` y donnent
un ecart de 0.0. Le codebook MNIST vaut toujours 5 945 408 octets, identique a l'entree du 17:45.
Les chiffres de `06_gan.ipynb` restent donc valides.

# 2026-07-17 12:20, l'AutoEncoder de reference n'avait pas converge

Premiere mesure de la tache 2, avec l'AE de reference entraine 15 epochs comme dans le notebook
MNIST : GAN 0.00706 de MSE contre AE 0.01164. **Le GAN battait l'AE en reconstruction**, l'inverse
de MNIST. Resultat suffisamment surprenant pour etre verifie avant d'etre ecrit.

Il etait faux. L'AE n'avait simplement pas fini d'apprendre a 15 epochs :

| budget AE | MSE | 3 dernieres loss train |
|---|---|---|
| 15 ep | 0.01164 | 0.01274, 0.01232, 0.01190 |
| 40 ep | 0.00813 | 0.00835, 0.00832, 0.00826 |
| 80 ep | 0.00624 | 0.00656, 0.00646, 0.00640 |
| 150 ep | 0.00477 | 0.00491, 0.00489, 0.00488 |

A 150 epochs l'AE est a 0.00477 et bat nettement le GAN (0.00706). La conclusion de MNIST tient
donc, et le "GAN meilleur que l'AE" n'etait qu'un artefact de budget inegal : 250 epochs pour le
GAN contre 15 pour sa reference. shapes est plus dur que MNIST, 15 epochs y suffisaient, ici non.
La lecon est generale : une reference sous-entrainee n'est pas une reference, elle est un
faire-valoir. `EPOCHS_AE = 150` dans le notebook, avec la justification en commentaire.

# 2026-07-17 13:00, generation ratee mais inversion reussie, et une hypothese qui tombe

Asymetrie inattendue entre les taches 1 et 2, sur le **meme** `G` :

- `generate()`, `z ~ N(0, I)` : taches de couleur parasitees, aucune forme lisible.
- `invert()` puis `decode()` : losanges, cercles, triangles, etoiles, croix, avec la bonne couleur
  et la bonne position. MSE 0.00706, comparable a une PCA a dimension latente egale.

Donc `G` **sait** produire des formes nettes, mais le tirage dans le prior ne les atteint pas.

Hypothese immediate : l'inversion va chercher des `z` de norme atypique, hors de la zone ou le
prior a sa masse. **Mesure, et l'hypothese tombe** :

| | mediane \|\|z\|\| | min | max | ecart-type par coordonnee |
|---|---|---|---|---|
| z issus de invert | 9.42 | 6.85 | 13.77 | 0.953 |
| z ~ N(0, I) | 9.94 | 7.68 | 12.41 | 0.998 |

Les codes trouves par inversion sont **dans** le prior : 0.5 % seulement depassent la norme max d'un
echantillon du prior. Ce n'est donc pas une sortie du typical set. Renormaliser un `z` inverse a la
norme mediane du prior degrade d'ailleurs la MSE de 0.0048 a 0.0076, ce qui confirme que la
position compte, pas le rayon.

Ce que disent les scores du critique (l'ecart entre moyennes, le score n'etant pas une probabilite) :
vraies -7.18, `G(z inverse)` -9.48, `G(z ~ prior)` -10.97. Les reconstructions sont jugees plus
credibles que les tirages du prior, et les deux restent sous les vraies.

Interpretation prudente : en dimension 100, une region peut avoir les memes statistiques marginales
(norme, ecart-type par coordonnee) que le prior tout en etant de masse negligeable. La variete des
formes est atteignable par descente de gradient mais reste rarement echantillonnee. C'est
exactement ce que dit la distance de Wasserstein qui plateau a 3.87 et non a 0 : la distribution
poussee par `G` ne coincide pas avec celle des donnees. Je m'arrete la, le mecanisme precis n'est
pas mesure et je ne l'inventerai pas.

Piste ecartee faute de mesure, pas faute d'idee : un `z` sample de norme plus faible (truncation
trick) irait tester la meme question sous un autre angle.

Detour a ne pas refaire : j'ai voulu quantifier la "nettete" par l'energie des gradients spatiaux.
La metrique note les **taches generees plus nettes (0.054) que les reconstructions (0.034)**, a
egalite avec les vraies images (0.054). Elle mesure en fait le bruit poivre-et-sel du fond, pas la
geometrie. Metrique jetee : mieux vaut pas de chiffre qu'un chiffre qui mesure autre chose que ce
qu'il annonce.

# 2026-07-17 14:30, un notebook vanilla assume sur shapes, et l'inversion qui tranche

`06_gan_shapes.ipynb` couvre le chemin complet MLP -> conv -> WGAN-GP. Le WGAN-GP y est justifie
par une pathologie mesuree, mais il empile beaucoup de concepts (critique, penalite de gradient,
LayerNorm vs BatchNorm, n_critic) pour un notebook qui doit rester explicable de bout en bout.
Ecriture d'un pendant strictement vanilla, `06_gan_shapes_vanilla.ipynb` : la classe `GAN` de
`src/gan.py` telle quelle, `data_dim` 784 -> 3072, rien d'autre. Aucune modification de `src/`,
aucune modification du notebook existant. Les deux se completent, ils ne se remplacent pas.

Reproduction exacte des chiffres du 09:30 (meme lot, effectifs `[1610, 1691, 1686, 1656, 1726,
1631]`, identiques a `01_kmeans_shapes.ipynb`) : accuracy de `D` 0.549, variance de `G` 0.123,
loss D 1.368 / loss G 0.743, ecart-type inter-images 0.1595 contre 0.1765 au reel. Images : taches.

**Le fait nouveau est sur la tache 2, et il est plus tranchant que tout le reste.** Le 09:30 ne
mesurait que la generation. Ici l'inversion a ete mesuree sur le `G` MLP, et **elle echoue de la
meme facon** : les reconstructions retrouvent la couleur dominante, la taille et la position
approximative, jamais la forme. Un losange bleu devient une tache bleue.

| | GAN (MLP, invert) | AutoEncoder (150 ep) |
|---|---|---|
| MSE | 0.01866 | 0.00477 |
| codebook | 15 326 208 o | 13 411 860 o |
| ratio (N = 10000) | 6.36 | 7.06 |

Pourquoi c'est la mesure qui compte : l'inversion est une descente de gradient de 300 pas **dediee a
une seule image**. Si un `z` produisant cette croix existait, elle aurait toutes les chances de le
trouver. Elle ne le trouve pas. La tache 1 pouvait laisser croire a un probleme d'echantillonnage du
prior ; la tache 2 elimine cette hypothese. Le `G` MLP **ne sait pas produire une geometrie**, quel
que soit `z`. Mode dropping total sur la forme, la ou MNIST n'en montrait qu'un partiel.

Ce point est le contraste net avec le WGAN-GP convolutif du 13:00 : la-bas `generate` echouait mais
`invert` retrouvait les formes (MSE 0.00706), d'ou l'asymetrie "meilleur decodeur que generateur".
Cette asymetrie est un **fait du convolutif, pas du GAN en general**. Avec un `G` MLP les deux
taches echouent ensemble, ce qui isole proprement la cause : le prior spatial. Le detour par la
distinction generation / inversion n'a de sens qu'une fois les convolutions en place.

Le GAN vanilla est aussi **domine sur les deux axes** de la compression : plus mauvais en MSE (3.9x)
et codebook plus lourd (15.3 Mo contre 13.4), donc ratio inferieur. Aucun regime de N ne le sauve,
contrairement a MNIST. Seuil de rentabilite 1289 images, plafond 30.72.

Projection : kNN 0.249 sur les features de `D` contre 0.245 sur les pixels bruts, hasard a 0.167.
L'ecart est dans le bruit. A noter, sur MNIST les features faisaient legerement **moins** bien
(0.430 contre 0.441), ici legerement mieux : dans les deux cas le detour n'apporte rien, et il ne
faut pas surinterpreter le signe de l'ecart.

Verdict du notebook vanilla : **echec sur les trois taches**, une seule cause, mesuree deux fois.
Resultat negatif, il reste. Il a l'avantage d'etre plus simple a defendre que le chemin WGAN-GP :
une seule pathologie, un seul diagnostic (`D` aveugle faute de voisinage), et la correction (prior
spatial) renvoyee a `06_gan_shapes.ipynb`.

