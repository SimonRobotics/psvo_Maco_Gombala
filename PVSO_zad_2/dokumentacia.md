# Kalibrácia kamery

Kalibrácia kamery slúži na získanie vnútorných parametrov kamery a koeficientov skreslenia objektívu. Tieto parametre sa následne používajú na odstránenie deformácie obrazu (undistortion) a umožňujú presnejšie spracovanie obrazu a meranie objektov.

Kalibrácia bola realizovaná pomocou šachovnicového vzoru a knižnice **OpenCV**.

## Postup

1. Načítajú sa snímky šachovnice z priečinka.
2. V každom obrázku sa detegujú vnútorné rohy šachovnice pomocou `cv2.findChessboardCorners()`.
3. Poloha rohov sa spresní pomocou `cv2.cornerSubPix()`.
4. Zo známych 3D bodov šachovnice a ich 2D projekcie v obraze sa vypočítajú parametre kamery pomocou `cv2.calibrateCamera()`.

Výsledkom kalibrácie sú:

* **camera matrix** – intrinzické parametre kamery
* **distortion coefficients** – koeficienty skreslenia objektívu
* **rotation a translation vectors** pre jednotlivé kalibračné snímky

Tieto parametre sa uložia do súboru `calibration_data.npz`, aby ich bolo možné použiť pri ďalšom spracovaní obrazu.

## Overenie kalibrácie

Správnosť kalibrácie sa overí odstránením skreslenia obrazu pomocou funkcie `cv2.undistort()`. Porovnaním pôvodného a opraveného obrazu je možné vizuálne skontrolovať, že línie v obraze sú po korekcii rovné.

![Porovnanie pôvodného a opraveného obrazu](Photos/undistorted.png)

## Poznámka

Ak je známa reálna veľkosť políčka šachovnice, je možné ju zahrnúť do modelu bodov šachovnice. V takom prípade je možné neskôr použiť kalibráciu aj na odhad reálnych rozmerov objektov v centimetroch.

# Detekcia tvarov

Táto časť programu slúži na detekciu základných geometrických tvarov v obraze z kamery. Pred samotnou detekciou sa načítajú parametre kalibrácie kamery, aby bolo možné odstrániť skreslenie obrazu.

Program pracuje v reálnom čase so streamom z kamery Ximea. Každý snímok sa načíta, zmenší, opraví pomocou kalibrácie a prevedie na odtiene sivej. Takto pripravený obraz sa ďalej používa na detekciu kruhov aj polygonálnych tvarov.

## Detekcia kruhov

Kruhy sa detegujú pomocou Houghovej transformácie `cv2.HoughCircles()`. Parametre detekcie, ako napríklad minimálna vzdialenosť kruhov, prahové hodnoty alebo rozsah polomerov, sa nastavujú pomocou trackbarov. Vďaka tomu je možné ladenie priamo počas behu programu.

Pred detekciou sa obraz jemne vyhladí filtrom `medianBlur`, aby sa znížil šum. Nájdené kruhy sa následne vykreslia do výstupného obrazu.

## Detekcia trojuholníkov, obdĺžnikov a štvorcov

Ostatné tvary sa detegujú pomocou kontúr. Najprv sa z obrazu vypočítajú hrany pomocou `Canny`, potom sa nájdu kontúry objektov. Každá kontúra sa aproximuje polygonálnou krivkou pomocou `cv2.approxPolyDP()`.

Typ tvaru sa určuje podľa počtu vrcholov:

* 3 vrcholy – trojuholník,
* 4 vrcholy – štvorec alebo obdĺžnik.

Pri štvoruholníkoch sa rozlíšenie medzi štvorcom a obdĺžnikom robí pomocou pomeru strán ohraničujúceho obdĺžnika.

## Ťažisko objektu

Pre každý detegovaný polygon sa vypočíta ťažisko pomocou obrazových momentov `cv2.moments()`. Ťažisko sa potom zobrazí vo výstupnom obraze spolu s jeho súradnicami.

## Výstup programu

Program zobrazuje viacero pomocných okien:

* pôvodný obraz s vyznačenými tvarmi,
* sivotónový obraz,
* rozmazaný obraz,
* hrany pre kruhy,
* hrany pre polygonálnu detekciu.

Takýto výstup umožňuje jednoduchšie ladenie parametrov a kontrolu správnosti detekcie.

![Detekcia tvarov](Photos/shapeDetection.png)

# Detekcia farby (HSV segmentácia)

Tento program slúži na jednoduchú detekciu farebných oblastí v obraze z kamery. Obraz z kamery sa najprv načíta a prevedie do farebného priestoru **HSV**, ktorý je vhodnejší na prácu s farbami než klasický RGB/BGR.

Používateľ môže pomocou posuvníkov (trackbarov) nastavovať dolné a horné hranice pre zložky **Hue**, **Saturation** a **Value**. Tieto hodnoty určujú, ktorá farba bude v obraze detegovaná.

Na základe zvolených hraníc sa vytvorí **binárna maska**, ktorá označuje pixely patriace do zvoleného farebného rozsahu. Táto maska sa potom použije na zvýraznenie detegovaných oblastí v obraze.

Výsledkom programu je:

* pôvodný obraz z kamery,
* binárna maska detegovanej farby,
* výsledný obraz, v ktorom sú detegované pixely zvýraznené.

![HSV segmentácia](Photos/colorFilter.png)

Program umožňuje interaktívne nastaviť vhodný rozsah HSV hodnôt pre konkrétnu farbu a je možné ho použiť napríklad na ladenie farebnej segmentácie pre ďalšie spracovanie obrazu.
