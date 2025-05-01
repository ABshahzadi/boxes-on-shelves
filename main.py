from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import itertools, random, warnings, time, os

from typing import List, Tuple
from multiprocessing import get_context, cpu_count
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

# ---------- Global evaluator function ----------
def evaluate_individual_with_data(args):
    ind, idx2prod, products, bays, shelves = args
    return decode_and_eval(ind, idx2prod, products, bays, shelves)

# ---------- 1) Data classes ----------
@dataclass
class Product:
    fam: str; qty: int; l: int; w: int; h: int
    def orientations(self):
        for dims in set(itertools.permutations((self.l, self.w, self.h))):
            yield dims

@dataclass
class Shelf:
    sid: int; thick: int; pos: int
    top_gap: int; left_gap: int; inter_gap: int; right_gap: int
    boxes: list = field(default_factory=list)

@dataclass
class Bay:
    W: int; H: int; D: int; avail_H: int
    shelves: List[Shelf] = field(default_factory=list)

# ---------- 2) File parsers ----------
def _clean_rows(fp):
    for ln in fp:
        ln = ln.strip()
        if ln and not ln.startswith('#'):
            yield ln.split()

def parse_products(path: Path) -> List[Product]:
    out = []
    with path.open() as f:
        for fam, qty, l, w, h in _clean_rows(f):
            out.append(Product(fam, int(qty), int(l), int(w), int(h)))
    return out

def parse_bays(path: Path) -> List[Bay]:
    bays = []
    with path.open() as f:
        for W, H, D, A in _clean_rows(f):
            bays.append(Bay(int(W), int(H), int(D), int(A)))
    return bays

def parse_shelves(path: Path) -> List[Shelf]:
    sh = []
    with path.open() as f:
        for sid, t, pos, tg, lg, ig, rg in _clean_rows(f):
            sh.append(Shelf(int(sid), int(t), int(pos),
                            int(tg), int(lg), int(ig), int(rg)))
    sh.sort(key=lambda s: s.pos)
    return sh

# ---------- 3) Decoder / Evaluator ----------
PENALTY = 10_000

def decode_and_eval(order: List[int], idx2prod: List[int],
                    prods: List[Product], bays: List[Bay], shelves: List[Shelf]) -> Tuple[int, int]:
    bays_local = [Bay(b.W, b.H, b.D, b.avail_H,
                      [Shelf(**asdict(s)) for s in shelves])
                  for b in bays]

    def place_box(dim):
        L, W, H = dim
        for bay in bays_local:
            for sh in bay.shelves:
                if H > sh.pos - sh.thick:
                    continue
                x0 = sh.left_gap if not sh.boxes else \
                     max(x + w for x, w in sh.boxes) + sh.inter_gap
                if x0 + W + sh.right_gap <= bay.W:
                    sh.boxes.append((x0, W))
                    return True
        return False

    for inst in order:
        p = prods[idx2prod[inst]]
        for orient in p.orientations():
            if orient[1] > bays_local[0].D:
                continue
            if place_box(orient):
                break
        else:
            return (PENALTY,) * 2

    used_shelves = sum(1 for bay in bays_local for sh in bay.shelves if sh.boxes)
    free_space = sum(bay.W - sh.right_gap - max(x + w for x, w in sh.boxes)
                     for bay in bays_local for sh in bay.shelves if sh.boxes)
    return used_shelves, free_space

# ---------- 4) Main GA ----------
def run_ga(datadir: Path, bay_file='bay2.txt', *, ngen=30, pop_size=120,
           cxpb=0.7, mutpb=0.2, n_jobs: int | None = None, seed=42,
           plot: bool = False, save_png: bool = False):

    random.seed(seed)
    products = parse_products(datadir / 'products.txt')
    bays = parse_bays(datadir / bay_file)
    shelves = parse_shelves(datadir / 'shelves.txt')

    print(f"Loaded: {len(products)} product types, "
          f"{sum(p.qty for p in products)} boxes total")

    ctx = get_context('spawn')
    n_cpu = n_jobs or cpu_count()

    idx2prod = [pid for pid, p in enumerate(products) for _ in range(p.qty)]
    n_items = len(idx2prod)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if 'FitnessMin' in creator.__dict__:
            del creator.FitnessMin
        if 'Individual' in creator.__dict__:
            del creator.Individual
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMin)

    tb = base.Toolbox()
    tb.register('individual', tools.initIterate,
                creator.Individual,
                lambda: random.sample(range(n_items), n_items))
    tb.register('population', tools.initRepeat, list, tb.individual)
    tb.register('mate', tools.cxPartialyMatched)
    tb.register('mutate', tools.mutShuffleIndexes, indpb=0.05)
    tb.register('select', tools.selNSGA2)

    hist_s, hist_f = [], []
    archive: list[Tuple[int, int]] = []

    with ctx.Pool(n_cpu) as pool:
        def parallel_eval(pop):
            args = [(ind, idx2prod, products, bays, shelves) for ind in pop]
            return pool.map(evaluate_individual_with_data, args)

        pop = tb.population(pop_size)
        for fit, ind in zip(parallel_eval(pop), pop):
            ind.fitness.values = fit
        pop = tb.select(pop, len(pop))

        for gen in range(1, ngen + 1):
            t0 = time.perf_counter()
            offspring = algorithms.varAnd(pop, tb, cxpb, mutpb)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for fit, ind in zip(parallel_eval(invalid), invalid):
                ind.fitness.values = fit

            pop = tb.select(pop + offspring, pop_size)
            best = tools.selBest(pop, 1)[0]
            s, f = best.fitness.values
            hist_s.append(s); hist_f.append(f)
            archive.extend(ind.fitness.values for ind in pop)

            if gen % 5 == 0 or gen == ngen:
                print(f"Gen {gen:2d} | best shelves={s:.0f}, "
                      f"empty={f:.0f} | {time.perf_counter() - t0:.1f}s")

        print("=== Finished ===")
        print(f"Best result: shelves={hist_s[-1]:.0f}, empty={hist_f[-1]:.0f}")

    if plot or save_png:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].plot(hist_s); axs[0, 0].set_title('Shelf Count Over Generations')
        axs[0, 1].plot(hist_f, color='tab:orange'); axs[0, 1].set_title('Empty Space Over Generations')

        x = [t[0] for t in archive]; y = [t[1] for t in archive]
        axs[1, 0].scatter(x, y, s=8, alpha=0.4)
        axs[1, 0].set_title('Pareto Front (Archive)')

        frontier = sorted({(s, f) for s, f in archive if s < PENALTY})
        ef_x, ef_y = zip(*frontier)
        axs[1, 1].plot(ef_x, ef_y, '-o', ms=3, alpha=0.6, label='Efficient Frontier')
        axs[1, 1].set_title('Efficient Frontier')
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.grid(True)
        plt.tight_layout()
        if save_png:
            try:
                output_dir = Path('results')
                output_dir.mkdir(exist_ok=True)
                fig.savefig(output_dir / 'progress_plots.png', dpi=300)
                print('\n✅ Plot saved as: results/progress_plots.png')
            except Exception as e:
                print(f"\n❌ Failed to save plot: {e}")
        if plot:
            plt.show()

    return hist_s, hist_f, archive

# ------------- Direct execution -------------
if __name__ == '__main__':
    run_ga(Path('data'),
           bay_file='bay2.txt',
           ngen=30, pop_size=120,
           plot=True, save_png=True)
