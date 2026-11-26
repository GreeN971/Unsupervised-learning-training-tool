# Vzorce pro odhad velikosti modelu

## Základní výpočet parametrů pro sekvenční (MLP) v bajtech

- $L$ je počet skrytých vrstev.
- $n_{in}$ je počet vstupů sítě. Při jednom sloupci dat je roven 1, při více sloupcích je roven dvojnásobku počtu sloupců, protože ke každé hodnotě je přidán příznak její přítomnosti.
- $n_i$ je počet neuronů v $i$-té vrstvě. Pro $i = 1 \ldots L$ jde o skryté vrstvy.
- $n_{L+1}$ je výstupní vrstva, která má vždy 1 neuron (Dense(1)). Proto suma u biasů končí až na $L+1$.

$$W = n_{in}\,n_1 + \sum_{i=1}^{L} n_i\,n_{i+1}$$

*Obrázek 5 Výpočet vah MLP [32]*

$$B = \sum_{i=1}^{L+1} n_i$$

*Obrázek 6 Výpočet biasů MLP [32]*

## Základní výpočet parametrů pro funkční (reziduální) v bajtech

- $N$ je počet reziduálních bloků.
- $j$ je index bloku, $j = 1 \ldots N$. Všechny bloky jsou stejné, takže člen v sumě na $j$ nezávisí.
- $D$ je počet Dense vrstev v každém bloku, přičemž $D \geq 2$. První vrstva bloku rozšiřuje šířku z $d$ na $h$, dalších $D - 2$ vrstev šířku $h$ zachovává a poslední vrstva ji vrací zpět na $d$.
- $d$ je šířka modelu, tedy počet neuronů vstupní Dense vrstvy (stem) a poslední Dense vrstvy v každém bloku. Musí být všude stejná, aby šel výstup bloku sečíst s jeho vstupem.
- $h$ je počet neuronů vnitřních Dense vrstev bloku.
- $k$ je expanzní faktor, který určuje $h$ vůči $d$.
- $n_{in}$ je počet vstupů sítě. Při jednom sloupci dat je roven 1, při více sloupcích je roven dvojnásobku počtu sloupců.

$$W = n_{in}\,d + \sum_{j=1}^{N}\left[\,2\,d\,h + (D-2)\,h^2\,\right] + d$$

*Obrázek 7 Výpočet vah Reziduální*

$$B = d + \sum_{j=1}^{N}\left[\,(D-1)\,h + d\,\right] + 1$$

*Obrázek 8 Výpočet biasů Reziduální*

$$h = k \cdot d$$

*Obrázek 9 Výpočet vnitřní šířky*

Součin $k \cdot d$ musí být celé číslo alespoň 1, aby měly vnitřní vrstvy celý počet neuronů. Pro $D = 2$ odpovídá blok klasickému uspořádání Dense–ReLU–Dense a člen $(D - 2)\,h^2$ je nulový.

## Finální vzorce pro float32

Finální vzorec pro výpočet velikosti MLP sítě v bytech není plně doložen literaturou. Jeho základ ano, ale přípočet vrstev jako $C_{layer}$ není, proto hypotéza bude ověřena v praktické části. $F$ představuje počet bajtů na jeden parametr, což pro formát float32 znamená 4 bajty, neboli počet bitů jednoho parametru dělený 8. Člen $L + 1$ odpovídá počtu Dense vrstev včetně výstupní.

$$S_{MLP} = F\,(W + B) + C_{layer}\,(L + 1) + C_{fixed}$$

*Obrázek 10 Finální vzorec pro výpočet MLP sítě*

Finální vzorec pro výpočet velikosti reziduální sítě v bytech není plně doložen literaturou. Sdílí některé faktory s MLP, ale zdaleka ne všechny, takže reziduální síť je o něco méně podložena a hypotéza stanovená tímto vzorcem bude také ověřena v praktické části. Počet Dense vrstev reziduální sítě je $N \cdot D + 2$, tedy vrstvy všech bloků, vstupní vrstva (stem) a výstupní vrstva. Přemosťovací spojení je specifické pro funkční topologii a přidává novou konstantu zastupující tato spojení. V rámci práce je jí dáno označení $C_{add}$ a má hodnotu 140 bajtů.

$$S_{RES} = F\,(W + B) + C_{layer}\,(N D + 2) + C_{add}\,N + C_{fixed}$$

*Obrázek 11 Finální vzorec pro výpočet Reziduální sítě*

## Finální vzorce pro int8

Pro kvantizovaný model int8 nelze vzorce převzít pouhou změnou $F$ na 1. Váhy sice zabírají 1 bajt, biasy však zůstávají 32bitové a kvantizace po kanálech ukládá pro každý neuron navíc měřítko a nulový bod. Vzorce proto pracují s váhami a biasy odděleně:

- $F_W$ je počet bajtů na jednu váhu, $F_W = 1$.
- $F_B$ je počet bajtů na jeden bias, $F_B = 4$.
- $C_q$ jsou kvantizační metadata na jeden neuron, $C_q = 24$ bajtů. Počet neuronů odpovídá počtu biasů, proto $C_q$ násobí $B$.
- $C'_{layer}$, $C'_{add}$ a $C'_{fixed}$ jsou konstanty se stejným významem jako u float32, změřené na modelech int8: $C'_{layer} = 509$, $C'_{add} = 160$ a $C'_{fixed} = 724$ bajtů.

$$S_{MLP,int8} = F_W\,W + (F_B + C_q)\,B + C'_{layer}\,(L + 1) + C'_{fixed}$$

*Obrázek 12 Finální vzorec pro výpočet MLP sítě int8*

$$S_{RES,int8} = F_W\,W + (F_B + C_q)\,B + C'_{layer}\,(N D + 2) + C'_{add}\,N + C'_{fixed}$$

*Obrázek 13 Finální vzorec pro výpočet Reziduální sítě int8*

---

## Poznámky k úpravám (není součástí textu práce)

**Změny oproti původní verzi:**

- **W u MLP:** doplněna vstupní vrstva $n_{in}\,n_1$.
- **Výstupní vrstva MLP:** označení opraveno z $n_{i+1}$ na $n_{L+1}$.
- **Reziduální síť:** doplněna proměnná $D$, člen $(D-2)\,h^2$ ve $W$, člen $(D-1)\,h$ v $B$ a počet vrstev $N D + 2$ místo $2N + 2$.
- **Vnitřní šířka:** $h = \max(1, \text{round}(k\,d))$ nahrazeno $h = k \cdot d$, protože aplikace zaokrouhlení nepovoluje.
- **Popisy proměnných:** „skrytý faktor“ změněn na „expanzní faktor“, $d$ je „poslední“ (ne „druhá“) Dense vrstva bloku a $n_{in}$ je definováno jako počet vstupů sítě.
- **Překlepy:** opraveno „je je jí“, „Překlenovací“ a velké R v „Reziduální“ uprostřed věty.
- **Nová část int8** s vlastními vzorci.

**Nesoulad s aplikací:**

1. **Float32 MLP v kódu** ([app/pages/topology.py](app/pages/topology.py), `estimate_mlp_float32_bytes`) zatím nepočítá člen $n_{in}\,n_1$, takže neodpovídá Obrázku 5. Konstanty 450 a 650 byly změřeny bez tohoto členu.
2. **Označení proměnných:** záložka topologie používá S, B, D a h (šířka, počet bloků, vrstvy v bloku, expanzní faktor). V textu jsou to $d$, $N$, $D$ a $k$, přičemž $h$ znamená vnitřní šířku.
